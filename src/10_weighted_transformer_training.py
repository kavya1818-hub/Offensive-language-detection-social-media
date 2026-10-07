# ============================================================
# STEP 10 - CLASS-WEIGHTED XLM-R TRANSFORMER TRAINING
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "xlm-roberta-base"

TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"

OUTPUT_DIR = "models/xlm_roberta_weighted"
CHECKPOINT_DIR = "models/xlm_roberta_weighted_checkpoints"

RESULTS_DIR = "results/weighted_transformer"

MAX_LENGTH = 64

# CPU-friendly settings
TRAINING_SAMPLES = 5000
VALIDATION_SAMPLES = 1000

BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5

SEED = 42


# ============================================================
# DEVICE
# ============================================================

print("=" * 60)
print("STEP 10 - CLASS-WEIGHTED XLM-R TRANSFORMER TRAINING")
print("=" * 60)

print("\nCOMPUTING DEVICE")
print("-" * 60)

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("GPU available.")
    print("Device:", torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("GPU not available.")
    print("Using CPU-optimized weighted transformer experiment.")
    print("Device: cpu")


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nTraining records :", len(train_df))
print("Testing records  :", len(test_df))

# Detect columns
if "model_text" in train_df.columns:
    TEXT_COLUMN = "model_text"
elif "normalized_text" in train_df.columns:
    TEXT_COLUMN = "normalized_text"
elif "tweet" in train_df.columns:
    TEXT_COLUMN = "tweet"
else:
    raise ValueError("No text column found.")

if "label" in train_df.columns:
    LABEL_COLUMN = "label"
else:
    raise ValueError("Label column not found.")

print("\nText column :", TEXT_COLUMN)
print("Label column:", LABEL_COLUMN)


# ============================================================
# CLASS MAPPING
# ============================================================

classes = sorted(train_df[LABEL_COLUMN].unique())

label2id = {
    label: idx
    for idx, label in enumerate(classes)
}

id2label = {
    idx: label
    for label, idx in label2id.items()
}

print("\nNumber of classes:", len(classes))

print("\nClasses:")
for idx, label in id2label.items():
    print(f"- {label}")


# ============================================================
# CLEAN DATA
# ============================================================

train_df = train_df.dropna(
    subset=[TEXT_COLUMN, LABEL_COLUMN]
).copy()

test_df = test_df.dropna(
    subset=[TEXT_COLUMN, LABEL_COLUMN]
).copy()

train_df[TEXT_COLUMN] = train_df[TEXT_COLUMN].astype(str)
test_df[TEXT_COLUMN] = test_df[TEXT_COLUMN].astype(str)

train_df["labels"] = train_df[LABEL_COLUMN].map(label2id)
test_df["labels"] = test_df[LABEL_COLUMN].map(label2id)


# ============================================================
# CREATE STRATIFIED TRANSFORMER SUBSET
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRANSFORMER TRAINING SUBSET")
print("=" * 60)

print("\nFull training dataset :", len(train_df))

# Stratified sampling
transformer_train_parts = []

for label in classes:

    class_data = train_df[
        train_df[LABEL_COLUMN] == label
    ]

    proportion = len(class_data) / len(train_df)

    sample_size = max(
        1,
        int(round(TRAINING_SAMPLES * proportion))
    )

    sample_size = min(sample_size, len(class_data))

    sampled = class_data.sample(
        n=sample_size,
        random_state=SEED
    )

    transformer_train_parts.append(sampled)

transformer_train = pd.concat(
    transformer_train_parts
).sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


# ============================================================
# VALIDATION SPLIT
# ============================================================

validation_parts = []

for label in classes:

    class_data = transformer_train[
        transformer_train[LABEL_COLUMN] == label
    ]

    proportion = len(class_data) / len(transformer_train)

    sample_size = max(
        1,
        int(round(VALIDATION_SAMPLES * proportion))
    )

    sample_size = min(
        sample_size,
        max(1, len(class_data) - 1)
    )

    sampled = class_data.sample(
        n=sample_size,
        random_state=SEED
    )

    validation_parts.append(sampled)


validation_df = pd.concat(
    validation_parts
).sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


# Remove validation records from training
validation_indices = set(validation_df.index)

# Re-create training subset safely using original row identity
transformer_train = transformer_train[
    ~transformer_train.index.isin(validation_indices)
].reset_index(drop=True)


print("\nTransformer training :", len(transformer_train))
print("Validation           :", len(validation_df))
print("Final testing        :", len(test_df))


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TRANSFORMER TRAINING DISTRIBUTION")
print("=" * 60)

print(
    transformer_train[LABEL_COLUMN].value_counts()
)

print("\nValidation Distribution:")

print(
    validation_df[LABEL_COLUMN].value_counts()
)

print("\nTesting Distribution:")

print(
    test_df[LABEL_COLUMN].value_counts()
)


# ============================================================
# IMPORTANT:
# USE FIXED ORIGINAL TEST SET
# ============================================================

final_test_df = test_df.copy()


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING CLASS WEIGHTS")
print("=" * 60)

class_counts = transformer_train["labels"].value_counts().sort_index()

total_samples = len(transformer_train)
num_classes = len(classes)

# Balanced class weights:
#
# weight_i = total_samples /
#            (number_of_classes * class_count_i)

class_weights = (
    total_samples /
    (num_classes * class_counts.values)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
)

print("\nClass counts:")

for idx, count in enumerate(class_counts.values):
    print(
        f"{id2label[idx]:25s}: {count}"
    )

print("\nCalculated class weights:")

for idx, weight in enumerate(class_weights):
    print(
        f"{id2label[idx]:25s}: {weight.item():.4f}"
    )


# ============================================================
# CONVERT TO HUGGINGFACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    transformer_train[
        [TEXT_COLUMN, "labels"]
    ],
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_df[
        [TEXT_COLUMN, "labels"]
    ],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    final_test_df[
        [TEXT_COLUMN, "labels"]
    ],
    preserve_index=False
)


# ============================================================
# TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_function(batch):

    return tokenizer(
        batch[TEXT_COLUMN],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

print("\nTokenization completed.")


# ============================================================
# FORMAT DATASET
# ============================================================

columns_to_keep = [
    "input_ids",
    "attention_mask",
    "labels"
]

train_dataset.set_format(
    type="torch",
    columns=columns_to_keep
)

validation_dataset.set_format(
    type="torch",
    columns=columns_to_keep
)

test_dataset.set_format(
    type="torch",
    columns=columns_to_keep
)


# ============================================================
# LOAD XLM-R
# ============================================================

print("\n" + "=" * 60)
print("LOADING XLM-R CLASSIFICATION MODEL")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id
)

print("\nModel loaded successfully.")


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1
    }


# ============================================================
# CUSTOM TRAINER WITH CLASS-WEIGHTED LOSS
# ============================================================

class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        # Move weights to same device as model
        weights = class_weights.to(
            logits.device
        )

        loss_function = torch.nn.CrossEntropyLoss(
            weight=weights
        )

        loss = loss_function(
            logits.view(-1, num_classes),
            labels.view(-1)
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

print("\n" + "=" * 60)
print("CONFIGURING WEIGHTED TRANSFORMER TRAINING")
print("=" * 60)

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,
    weight_decay=0.01,

    logging_steps=50,

    save_strategy="epoch",

    report_to="none",

    fp16=False,

    dataloader_pin_memory=False,

    seed=SEED
)


# ============================================================
# TRAINER
# ============================================================

trainer = WeightedTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    compute_metrics=compute_metrics
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("CLASS-WEIGHTED XLM-R TRAINING")
print("=" * 60)

print("\nTraining samples :", len(train_dataset))
print("Validation samples:", len(validation_dataset))
print("Epochs            :", EPOCHS)
print("Maximum length    :", MAX_LENGTH)
print("Batch size        :", BATCH_SIZE)

print("\nClass weighting is ENABLED.")

print("\nTraining started...\n")

start_time = time.time()

trainer.train()

training_time = time.time() - start_time

print("\nTraining completed.")

print(
    f"Training time: {training_time / 60:.2f} minutes"
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING WEIGHTED XLM-R MODEL")
print("=" * 60)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("\nModel saved:")
print(OUTPUT_DIR)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("WEIGHTED XLM-R VALIDATION EVALUATION")
print("=" * 60)

validation_results = trainer.evaluate(
    validation_dataset
)

for key, value in validation_results.items():

    if isinstance(value, float):
        print(
            f"{key}: {value:.4f}"
        )
    else:
        print(
            f"{key}: {value}"
        )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("WEIGHTED XLM-R FINAL TEST EVALUATION")
print("=" * 60)

test_results = trainer.evaluate(
    test_dataset
)

print("\nWEIGHTED XLM-R TEST RESULTS")

for key, value in test_results.items():

    if key.startswith("eval_"):

        metric_name = key.replace(
            "eval_",
            ""
        )

        if isinstance(value, float):

            print(
                f"{metric_name:20s}: {value:.4f}"
            )


# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("GENERATING TEST PREDICTIONS")
print("=" * 60)

prediction_output = trainer.predict(
    test_dataset
)

logits = prediction_output.predictions

predictions = np.argmax(
    logits,
    axis=1
)

true_labels = prediction_output.label_ids


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

from sklearn.metrics import classification_report

report = classification_report(
    true_labels,
    predictions,
    target_names=classes,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_path = os.path.join(
    RESULTS_DIR,
    "weighted_xlm_roberta_classification_report.csv"
)

report_df.to_csv(
    report_path
)

print("\nClassification report saved:")
print(report_path)


# ============================================================
# CONFUSION MATRIX
# ============================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    true_labels,
    predictions
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual_{label}"
        for label in classes
    ],
    columns=[
        f"Predicted_{label}"
        for label in classes
    ]
)

cm_path = os.path.join(
    RESULTS_DIR,
    "weighted_xlm_roberta_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)

print("Confusion matrix saved:")
print(cm_path)


# ============================================================
# PREDICTION FILE
# ============================================================

prediction_df = final_test_df.copy()

prediction_df["actual_label"] = [
    id2label[int(x)]
    for x in true_labels
]

prediction_df["predicted_label"] = [
    id2label[int(x)]
    for x in predictions
]

prediction_df["correct"] = (
    prediction_df["actual_label"]
    ==
    prediction_df["predicted_label"]
)

prediction_path = os.path.join(
    RESULTS_DIR,
    "weighted_xlm_roberta_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)

print("Prediction results saved:")
print(prediction_path)


# ============================================================
# ERROR ANALYSIS
# ============================================================

misclassified_df = prediction_df[
    prediction_df["correct"] == False
].copy()

misclassified_path = os.path.join(
    RESULTS_DIR,
    "weighted_xlm_roberta_misclassified.csv"
)

misclassified_df.to_csv(
    misclassified_path,
    index=False
)

print("\nMisclassified samples:", len(misclassified_df))

print("Misclassified samples saved:")
print(misclassified_path)


# ============================================================
# CLASS-WISE PERFORMANCE
# ============================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        true_labels,
        predictions,
        labels=list(range(num_classes)),
        zero_division=0
    )
)

class_results = []

for i in range(num_classes):

    class_results.append({

        "Class": id2label[i],

        "Actual": int(
            np.sum(true_labels == i)
        ),

        "Predicted": int(
            np.sum(predictions == i)
        ),

        "Correct": int(
            np.sum(
                (true_labels == i)
                &
                (predictions == i)
            )
        ),

        "Precision": precision[i],

        "Recall": recall[i],

        "F1": f1[i],

        "Class_Weight": float(
            class_weights[i]
        )
    })


class_results_df = pd.DataFrame(
    class_results
)

class_results_path = os.path.join(
    RESULTS_DIR,
    "weighted_xlm_roberta_class_wise_performance.csv"
)

class_results_df.to_csv(
    class_results_path,
    index=False
)

print("\nClass-wise performance:")
print(
    class_results_df.to_string(
        index=False
    )
)

print("\nClass-wise results saved:")
print(class_results_path)


# ============================================================
# FINAL SUMMARY
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predictions
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0
    )
)

summary = pd.DataFrame([{

    "Model": "XLM-R Weighted",

    "Accuracy": accuracy,

    "Macro_Precision": macro_precision,

    "Macro_Recall": macro_recall,

    "Macro_F1": macro_f1,

    "Weighted_Precision": weighted_precision,

    "Weighted_Recall": weighted_recall,

    "Weighted_F1": weighted_f1,

    "Training_Time_Minutes":
        training_time / 60,

    "Training_Samples":
        len(train_dataset),

    "Validation_Samples":
        len(validation_dataset),

    "Test_Samples":
        len(test_dataset),

    "Max_Length":
        MAX_LENGTH,

    "Epochs":
        EPOCHS

}])

summary_path = os.path.join(
    RESULTS_DIR,
    "weighted_xlm_roberta_results.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("STEP 10 - WEIGHTED TRANSFORMER MODELING COMPLETED")
print("=" * 60)

print("\nFINAL WEIGHTED XLM-R RESULTS:")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Macro F1  : {macro_f1:.4f}"
)

print(
    f"Weighted F1: {weighted_f1:.4f}"
)

print("\nGenerated files:")

print(
    f"- {OUTPUT_DIR}/"
)

print(
    f"- {CHECKPOINT_DIR}/"
)

print(
    f"- {summary_path}"
)

print(
    f"- {report_path}"
)

print(
    f"- {cm_path}"
)

print(
    f"- {prediction_path}"
)

print(
    f"- {misclassified_path}"
)

print(
    f"- {class_results_path}"
)

print("\n" + "=" * 60)