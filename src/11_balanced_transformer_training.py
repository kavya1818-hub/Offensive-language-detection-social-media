"""
STEP 11 - BALANCED XLM-R TRANSFORMER TRAINING

Purpose:
Train XLM-R on a more balanced training subset to improve
minority-class (especially Hate_Speech) detection.

This is intentionally different from Step 10:
- No aggressive inverse-frequency class weights
- Balanced training subset
- Macro F1 used as the main evaluation metric
- Checkpoints saved periodically
- CPU-friendly configuration
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "xlm-roberta-base"

TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"

MODEL_OUTPUT = "models/xlm_roberta_balanced"
CHECKPOINT_OUTPUT = "models/xlm_roberta_balanced_checkpoints"
RESULT_OUTPUT = "results/balanced_transformer"

MAX_LENGTH = 64

# CPU-friendly experiment
TRAINING_SAMPLES_PER_CLASS = 1000
VALIDATION_SAMPLES_PER_CLASS = 300

BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5

CHECKPOINT_STEPS = 100

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(MODEL_OUTPUT, exist_ok=True)
os.makedirs(CHECKPOINT_OUTPUT, exist_ok=True)
os.makedirs(RESULT_OUTPUT, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("STEP 11 - BALANCED XLM-R TRANSFORMER TRAINING")
print("=" * 60)


# ============================================================
# DEVICE
# ============================================================

print("\nCOMPUTING DEVICE")
print("-" * 60)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU available.")
    print("Using GPU.")
    print("Device:", device)
else:
    device = torch.device("cpu")
    print("GPU not available.")
    print("Using CPU-optimized balanced transformer experiment.")
    print("Device:", device)


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


# ============================================================
# DETECT COLUMNS
# ============================================================

if "model_text" in train_df.columns:
    TEXT_COLUMN = "model_text"
elif "normalized_text" in train_df.columns:
    TEXT_COLUMN = "normalized_text"
elif "tweet" in train_df.columns:
    TEXT_COLUMN = "tweet"
else:
    raise ValueError("No suitable text column found.")

if "label" in train_df.columns:
    LABEL_COLUMN = "label"
else:
    raise ValueError("Label column not found.")

print("\nText column :", TEXT_COLUMN)
print("Label column:", LABEL_COLUMN)


# ============================================================
# CLASS DEFINITIONS
# ============================================================

class_names = [
    "Hate_Speech",
    "Neither",
    "Offensive_Language"
]

label2id = {
    "Hate_Speech": 0,
    "Neither": 1,
    "Offensive_Language": 2
}

id2label = {
    0: "Hate_Speech",
    1: "Neither",
    2: "Offensive_Language"
}

print("\nNumber of classes:", len(class_names))

print("\nClasses:")
for name in class_names:
    print("-", name)


# ============================================================
# CLEAN DATA
# ============================================================

train_df = train_df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
test_df = test_df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()

train_df[TEXT_COLUMN] = train_df[TEXT_COLUMN].astype(str)
test_df[TEXT_COLUMN] = test_df[TEXT_COLUMN].astype(str)


# ============================================================
# ORIGINAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("ORIGINAL TRAINING DISTRIBUTION")
print("=" * 60)

print(train_df[LABEL_COLUMN].value_counts())


# ============================================================
# CREATE BALANCED TRAINING DATASET
# ============================================================

print("\n" + "=" * 60)
print("CREATING BALANCED TRAINING SUBSET")
print("=" * 60)

balanced_parts = []

for class_name in class_names:

    class_data = train_df[
        train_df[LABEL_COLUMN] == class_name
    ].copy()

    available = len(class_data)

    sample_count = min(
        TRAINING_SAMPLES_PER_CLASS,
        available
    )

    sampled = class_data.sample(
        n=sample_count,
        random_state=SEED
    )

    balanced_parts.append(sampled)

    print(
        f"{class_name:<22}: "
        f"available={available}, "
        f"selected={sample_count}"
    )


balanced_train_df = pd.concat(
    balanced_parts,
    ignore_index=True
)

balanced_train_df = balanced_train_df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


# ============================================================
# CREATE BALANCED VALIDATION SET
# ============================================================

validation_parts = []

for class_name in class_names:

    class_data = train_df[
        train_df[LABEL_COLUMN] == class_name
    ].copy()

    # Remove rows already selected for training
    remaining = class_data.drop(
        balanced_train_df.index,
        errors="ignore"
    )

    # Safer fallback
    if len(remaining) < VALIDATION_SAMPLES_PER_CLASS:
        remaining = class_data

    sample_count = min(
        VALIDATION_SAMPLES_PER_CLASS,
        len(remaining)
    )

    sampled = remaining.sample(
        n=sample_count,
        random_state=SEED + 1
    )

    validation_parts.append(sampled)


validation_df = pd.concat(
    validation_parts,
    ignore_index=True
)

validation_df = validation_df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


# ============================================================
# TEST DATA
# ============================================================

final_test_df = test_df.copy()


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("BALANCED DATASET SUMMARY")
print("=" * 60)

print("\nFull training dataset :", len(train_df))
print("Balanced training    :", len(balanced_train_df))
print("Validation            :", len(validation_df))
print("Final testing         :", len(final_test_df))

print("\nBalanced Training Distribution:")
print(
    balanced_train_df[LABEL_COLUMN].value_counts()
)

print("\nValidation Distribution:")
print(
    validation_df[LABEL_COLUMN].value_counts()
)

print("\nTesting Distribution:")
print(
    final_test_df[LABEL_COLUMN].value_counts()
)


# ============================================================
# CONVERT LABELS
# ============================================================

def convert_labels(df):
    df = df.copy()

    df["labels"] = df[LABEL_COLUMN].map(label2id)

    if df["labels"].isna().any():
        raise ValueError("Unknown label detected.")

    df["labels"] = df["labels"].astype(int)

    return df


balanced_train_df = convert_labels(balanced_train_df)
validation_df = convert_labels(validation_df)
final_test_df = convert_labels(final_test_df)


# ============================================================
# CREATE HF DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    balanced_train_df[
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


def tokenize_function(examples):

    return tokenizer(
        examples[TEXT_COLUMN],
        truncation=True,
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
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING XLM-R CLASSIFICATION MODEL")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
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
# TRAINING ARGUMENTS
# ============================================================

print("\n" + "=" * 60)
print("CONFIGURING BALANCED TRANSFORMER TRAINING")
print("=" * 60)

training_args = TrainingArguments(

    output_dir=CHECKPOINT_OUTPUT,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    weight_decay=0.01,

    logging_steps=50,

    eval_strategy="epoch",

    save_strategy="steps",
    save_steps=CHECKPOINT_STEPS,
    save_total_limit=3,

    load_best_model_at_end=False,

    report_to="none",

    fp16=False,

    dataloader_pin_memory=False,

    remove_unused_columns=True
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("BALANCED XLM-R TRAINING")
print("=" * 60)

print("\nTraining samples :", len(train_dataset))
print("Validation samples:", len(validation_dataset))
print("Epochs           :", EPOCHS)
print("Maximum length   :", MAX_LENGTH)
print("Batch size       :", BATCH_SIZE)
print("Learning rate    :", LEARNING_RATE)

print("\nBalanced training is ENABLED.")
print("Aggressive class weighting is DISABLED.")
print("Checkpoint interval:", CHECKPOINT_STEPS, "steps")

print("\nTraining started...")

start_time = time.time()

train_result = trainer.train()

training_time = (
    time.time() - start_time
) / 60

print("\nTraining completed.")

print(
    f"Training time: {training_time:.2f} minutes"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING BALANCED XLM-R MODEL")
print("=" * 60)

trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)

print("\nModel saved:")
print(MODEL_OUTPUT)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("BALANCED XLM-R VALIDATION EVALUATION")
print("=" * 60)

validation_results = trainer.evaluate(
    eval_dataset=validation_dataset
)

for key, value in validation_results.items():

    if isinstance(value, float):
        print(
            f"{key:<25}: {value:.4f}"
        )
    else:
        print(
            f"{key:<25}: {value}"
        )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("BALANCED XLM-R FINAL TEST EVALUATION")
print("=" * 60)

test_output = trainer.predict(
    test_dataset
)

test_predictions = np.argmax(
    test_output.predictions,
    axis=1
)

test_labels = np.array(
    test_dataset["labels"]
)


# ============================================================
# TEST METRICS
# ============================================================

accuracy = accuracy_score(
    test_labels,
    test_predictions
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        test_labels,
        test_predictions,
        average="macro",
        zero_division=0
    )
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        test_labels,
        test_predictions,
        average="weighted",
        zero_division=0
    )
)


print("\nBALANCED XLM-R TEST RESULTS")

print(
    f"Accuracy           : {accuracy:.4f}"
)

print(
    f"Macro Precision    : {macro_precision:.4f}"
)

print(
    f"Macro Recall       : {macro_recall:.4f}"
)

print(
    f"Macro F1           : {macro_f1:.4f}"
)

print(
    f"Weighted Precision : {weighted_precision:.4f}"
)

print(
    f"Weighted Recall    : {weighted_recall:.4f}"
)

print(
    f"Weighted F1       : {weighted_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    test_labels,
    test_predictions,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_path = os.path.join(
    RESULT_OUTPUT,
    "balanced_xlm_roberta_classification_report.csv"
)

report_df.to_csv(
    report_path
)

print(report_df)

print(
    "\nClassification report saved:"
)
print(report_path)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    test_labels,
    test_predictions
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_Hate_Speech",
        "Actual_Neither",
        "Actual_Offensive_Language"
    ],
    columns=[
        "Predicted_Hate_Speech",
        "Predicted_Neither",
        "Predicted_Offensive_Language"
    ]
)

print(cm_df)

cm_path = os.path.join(
    RESULT_OUTPUT,
    "balanced_xlm_roberta_confusion_matrix.csv"
)

cm_df.to_csv(cm_path)

print(
    "\nConfusion matrix saved:"
)
print(cm_path)


# ============================================================
# PREDICTIONS
# ============================================================

prediction_df = final_test_df.copy()

prediction_df["actual_label"] = (
    test_labels
)

prediction_df["predicted_label_id"] = (
    test_predictions
)

prediction_df["predicted_label"] = [
    id2label[x]
    for x in test_predictions
]

prediction_path = os.path.join(
    RESULT_OUTPUT,
    "balanced_xlm_roberta_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)

print(
    "\nPrediction results saved:"
)
print(prediction_path)


# ============================================================
# MISCLASSIFIED SAMPLES
# ============================================================

misclassified = prediction_df[
    prediction_df["actual_label"] !=
    prediction_df["predicted_label_id"]
].copy()

misclassified_path = os.path.join(
    RESULT_OUTPUT,
    "balanced_xlm_roberta_misclassified.csv"
)

misclassified.to_csv(
    misclassified_path,
    index=False
)

print(
    "\nMisclassified samples:",
    len(misclassified)
)

print(
    "Misclassified samples saved:"
)

print(misclassified_path)


# ============================================================
# CLASS-WISE PERFORMANCE
# ============================================================

class_rows = []

for class_id, class_name in id2label.items():

    actual_count = np.sum(
        test_labels == class_id
    )

    predicted_count = np.sum(
        test_predictions == class_id
    )

    correct_count = np.sum(
        (test_labels == class_id) &
        (test_predictions == class_id)
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            test_labels,
            test_predictions,
            labels=[class_id],
            average=None,
            zero_division=0
        )
    )

    class_rows.append({
        "Class": class_name,
        "Actual": actual_count,
        "Predicted": predicted_count,
        "Correct": correct_count,
        "Precision": precision[0],
        "Recall": recall[0],
        "F1": f1[0]
    })


classwise_df = pd.DataFrame(
    class_rows
)

print("\nClass-wise performance:")
print(classwise_df)


classwise_path = os.path.join(
    RESULT_OUTPUT,
    "balanced_xlm_roberta_class_wise_performance.csv"
)

classwise_df.to_csv(
    classwise_path,
    index=False
)

print(
    "\nClass-wise results saved:"
)

print(classwise_path)


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame([{

    "Model":
        "Balanced XLM-R",

    "Accuracy":
        accuracy,

    "Macro_Precision":
        macro_precision,

    "Macro_Recall":
        macro_recall,

    "Macro_F1":
        macro_f1,

    "Weighted_Precision":
        weighted_precision,

    "Weighted_Recall":
        weighted_recall,

    "Weighted_F1":
        weighted_f1,

    "Training_Samples":
        len(train_dataset),

    "Validation_Samples":
        len(validation_dataset),

    "Test_Samples":
        len(test_dataset),

    "Epochs":
        EPOCHS,

    "Max_Length":
        MAX_LENGTH,

    "Batch_Size":
        BATCH_SIZE,

    "Training_Time_Minutes":
        training_time
}])


summary_path = os.path.join(
    RESULT_OUTPUT,
    "balanced_xlm_roberta_results.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("STEP 11 - BALANCED TRANSFORMER MODELING COMPLETED")
print("=" * 60)

print("\nFINAL BALANCED XLM-R RESULTS:")

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
    f"- {MODEL_OUTPUT}/"
)

print(
    f"- {CHECKPOINT_OUTPUT}/"
)

print(
    f"- {RESULT_OUTPUT}/balanced_xlm_roberta_results.csv"
)

print(
    f"- {RESULT_OUTPUT}/balanced_xlm_roberta_classification_report.csv"
)

print(
    f"- {RESULT_OUTPUT}/balanced_xlm_roberta_confusion_matrix.csv"
)

print(
    f"- {RESULT_OUTPUT}/balanced_xlm_roberta_predictions.csv"
)

print(
    f"- {RESULT_OUTPUT}/balanced_xlm_roberta_misclassified.csv"
)

print(
    f"- {RESULT_OUTPUT}/balanced_xlm_roberta_class_wise_performance.csv"
)

print("\n" + "=" * 60)