# ============================================================
# STEP 8 - XLM-R TRANSFORMER MODELING
# Checkpoint + Resume Version
# ============================================================

import os
import time
import warnings
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
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "xlm-roberta-base"

TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"

MODEL_DIR = "models/xlm_roberta"
CHECKPOINT_DIR = "models/xlm_roberta_checkpoints"

RESULTS_DIR = "results/transformer"

MAX_LENGTH = 64
TRAINING_SAMPLES = 5000
VALIDATION_SAMPLES = 1000

NUM_EPOCHS = 1
BATCH_SIZE = 8

SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

print("=" * 60)
print("STEP 8 - XLM-R TRANSFORMER MODELING")
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
    print("Using CPU-optimized transformer experiment.")
    print("Device: cpu")

# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nDataset loaded successfully.")
print("Available training records :", len(train_df))
print("Testing records            :", len(test_df))

# ============================================================
# DETECT COLUMNS
# ============================================================

TEXT_COLUMN = None

for col in ["model_text", "normalized_text", "text", "tweet"]:
    if col in train_df.columns:
        TEXT_COLUMN = col
        break

if TEXT_COLUMN is None:
    raise ValueError("No suitable text column found.")

if "label" not in train_df.columns:
    raise ValueError("Label column not found.")

LABEL_COLUMN = "label"

print("\nText column detected :", TEXT_COLUMN)
print("Label column detected:", LABEL_COLUMN)

# ============================================================
# LABEL ENCODING
# ============================================================

classes = sorted(train_df[LABEL_COLUMN].unique())

label2id = {label: i for i, label in enumerate(classes)}
id2label = {i: label for label, i in label2id.items()}

train_df["label_id"] = train_df[LABEL_COLUMN].map(label2id)
test_df["label_id"] = test_df[LABEL_COLUMN].map(label2id)

print("\nNumber of classes:", len(classes))

print("\nClasses:")
for label in classes:
    print("-", label)

# ============================================================
# CREATE TRANSFORMER SUBSET
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRANSFORMER TRAINING SUBSET")
print("=" * 60)

# Shuffle reproducibly
train_shuffled = train_df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

# Use stratified sampling to preserve class distribution
from sklearn.model_selection import train_test_split

transformer_train, validation_df = train_test_split(
    train_shuffled,
    train_size=TRAINING_SAMPLES,
    test_size=VALIDATION_SAMPLES,
    stratify=train_shuffled[LABEL_COLUMN],
    random_state=SEED
)

transformer_train = transformer_train.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)

final_test = test_df.reset_index(drop=True)

print("\nFull training dataset :", len(train_df))
print("Transformer training  :", len(transformer_train))
print("Validation            :", len(validation_df))
print("Final testing         :", len(final_test))

print("\nTransformer Dataset Split:")
print("Training   :", len(transformer_train))
print("Validation :", len(validation_df))
print("Testing    :", len(final_test))

print("\nTransformer Training Distribution:")
print(transformer_train[LABEL_COLUMN].value_counts())

print("\nValidation Distribution:")
print(validation_df[LABEL_COLUMN].value_counts())

print("\nTesting Distribution:")
print(final_test[LABEL_COLUMN].value_counts())

# ============================================================
# CREATE HUGGINGFACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    transformer_train[[TEXT_COLUMN, "label_id"]],
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_df[[TEXT_COLUMN, "label_id"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    final_test[[TEXT_COLUMN, "label_id"]],
    preserve_index=False
)

# Rename label column
train_dataset = train_dataset.rename_column("label_id", "labels")
validation_dataset = validation_dataset.rename_column("label_id", "labels")
test_dataset = test_dataset.rename_column("label_id", "labels")

# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer:")
print(MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples[TEXT_COLUMN],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[TEXT_COLUMN]
)

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[TEXT_COLUMN]
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[TEXT_COLUMN]
)

print("\nTokenization completed.")

# ============================================================
# MODEL
# ============================================================

print("\nLoading XLM-R classification model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(classes),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

print("Model loaded successfully.")

# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted
    }

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

print("\nConfiguring transformer training...")

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=2e-5,

    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=100,

    save_total_limit=2,

    logging_steps=50,

    load_best_model_at_end=False,

    report_to="none",

    fp16=False,

    dataloader_pin_memory=False,

    seed=SEED
)

# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=validation_dataset,

    compute_metrics=compute_metrics
)

# ============================================================
# CHECKPOINT DETECTION
# ============================================================

print("\n" + "=" * 60)
print("TRANSFORMER TRAINING")
print("=" * 60)

print("\nTraining samples :", len(transformer_train))
print("Validation samples:", len(validation_df))
print("Epochs           :", NUM_EPOCHS)
print("Maximum length   :", MAX_LENGTH)
print("Batch size       :", BATCH_SIZE)

# Find latest checkpoint
checkpoints = []

if os.path.exists(CHECKPOINT_DIR):

    for item in os.listdir(CHECKPOINT_DIR):

        if item.startswith("checkpoint-"):

            path = os.path.join(CHECKPOINT_DIR, item)

            if os.path.isdir(path):
                try:
                    step = int(item.split("-")[1])
                    checkpoints.append((step, path))
                except:
                    pass

if checkpoints:

    checkpoints.sort(reverse=True)

    latest_checkpoint = checkpoints[0][1]

    print("\nExisting checkpoint detected:")
    print(latest_checkpoint)

    print("\nRESUMING TRAINING FROM CHECKPOINT...")

else:

    latest_checkpoint = None

    print("\nNo existing checkpoint found.")
    print("Starting training from the beginning.")

# ============================================================
# TRAIN
# ============================================================

start_time = time.time()

if latest_checkpoint:

    trainer.train(
        resume_from_checkpoint=latest_checkpoint
    )

else:

    trainer.train()

training_time = time.time() - start_time

print("\nTraining completed.")

print(
    "Training time: {:.2f} minutes".format(
        training_time / 60
    )
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\nSaving final XLM-R model...")

trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("Model saved:", MODEL_DIR)

# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION EVALUATION")
print("=" * 60)

validation_results = trainer.evaluate(
    validation_dataset
)

print("\nValidation Results:")

for key, value in validation_results.items():

    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# ============================================================
# FINAL TEST PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

prediction_output = trainer.predict(test_dataset)

test_predictions = np.argmax(
    prediction_output.predictions,
    axis=-1
)

test_labels = prediction_output.label_ids

# ============================================================
# TEST METRICS
# ============================================================

accuracy = accuracy_score(
    test_labels,
    test_predictions
)

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        test_labels,
        test_predictions,
        average="macro",
        zero_division=0
    )
)

precision_weighted, recall_weighted, f1_weighted, _ = (
    precision_recall_fscore_support(
        test_labels,
        test_predictions,
        average="weighted",
        zero_division=0
    )
)

print("\nXLM-R TEST RESULTS")

print("Accuracy          : {:.4f}".format(accuracy))
print("Macro Precision   : {:.4f}".format(precision_macro))
print("Macro Recall      : {:.4f}".format(recall_macro))
print("Macro F1          : {:.4f}".format(f1_macro))
print("Weighted Precision: {:.4f}".format(precision_weighted))
print("Weighted Recall   : {:.4f}".format(recall_weighted))
print("Weighted F1       : {:.4f}".format(f1_weighted))

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_labels,
    test_predictions,
    target_names=classes,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_file = os.path.join(
    RESULTS_DIR,
    "xlm_roberta_classification_report.csv"
)

report_df.to_csv(
    report_file,
    index=True
)

print("\nClassification report saved:")
print(report_file)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

cm_file = os.path.join(
    RESULTS_DIR,
    "xlm_roberta_confusion_matrix.csv"
)

cm_df.to_csv(cm_file)

print("Confusion matrix saved:")
print(cm_file)

# ============================================================
# TEST PREDICTIONS
# ============================================================

prediction_df = final_test.copy()

prediction_df["actual_label"] = [
    id2label[i]
    for i in test_labels
]

prediction_df["predicted_label"] = [
    id2label[i]
    for i in test_predictions
]

prediction_df["correct"] = (
    prediction_df["actual_label"]
    == prediction_df["predicted_label"]
)

prediction_file = os.path.join(
    RESULTS_DIR,
    "xlm_roberta_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print("Prediction results saved:")
print(prediction_file)

# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([{
    "Model": "XLM-R",
    "Training_Samples": len(transformer_train),
    "Validation_Samples": len(validation_df),
    "Testing_Samples": len(final_test),
    "Accuracy": accuracy,
    "Macro_Precision": precision_macro,
    "Macro_Recall": recall_macro,
    "Macro_F1": f1_macro,
    "Weighted_Precision": precision_weighted,
    "Weighted_Recall": recall_weighted,
    "Weighted_F1": f1_weighted,
    "Training_Time_Minutes": training_time / 60
}])

summary_file = os.path.join(
    RESULTS_DIR,
    "xlm_roberta_results.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print("\nSummary saved:")
print(summary_file)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("STEP 8 - TRANSFORMER MODELING COMPLETED")
print("=" * 60)

print("\nFinal XLM-R Results:")
print("Accuracy  : {:.4f}".format(accuracy))
print("Macro F1  : {:.4f}".format(f1_macro))

print("\nGenerated files:")
print("- models/xlm_roberta/")
print("- models/xlm_roberta_checkpoints/")
print("- results/transformer/xlm_roberta_results.csv")
print("- results/transformer/xlm_roberta_classification_report.csv")
print("- results/transformer/xlm_roberta_confusion_matrix.csv")
print("- results/transformer/xlm_roberta_predictions.csv")

print("\n" + "=" * 60)