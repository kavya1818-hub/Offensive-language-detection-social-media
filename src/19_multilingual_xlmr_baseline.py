"""
STEP 19 - CPU-FRIENDLY MULTILINGUAL XLM-R BASELINE

Purpose:
    Train a multilingual XLM-R baseline on Tamil, Malayalam and Kannada.

Important:
    - Uses the original train/dev/test splits.
    - Uses a stratified subset of the training set for CPU feasibility.
    - Uses the COMPLETE test set.
    - No class weighting.
    - No oversampling.
    - No focal loss.
    - This is the MULTILINGUAL BASELINE.
    - Checkpoints are saved during training.
    - Training automatically resumes from the latest checkpoint.
    - Compatible with older/newer Transformers Trainer APIs.
"""

import os
import sys
import random
import warnings
import inspect

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
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

SEED = 42

MODEL_NAME = "xlm-roberta-base"

DATA_PATH = "data/dravidian_clean/dravidian_multilingual_clean.csv"

OUTPUT_DIR = "results/multilingual_baseline"

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints"
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "final_model"
)

PREDICTION_DIR = os.path.join(
    OUTPUT_DIR,
    "predictions"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)

PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)


# CPU-FRIENDLY SETTINGS
TRAINING_SAMPLES = 6000
VALIDATION_SAMPLES = 1000

MAX_LENGTH = 96

EPOCHS = 1

BATCH_SIZE = 4

LEARNING_RATE = 2e-5

GRADIENT_ACCUMULATION = 2

SAVE_STEPS = 250

LOGGING_STEPS = 50

WARMUP_RATIO = 0.05


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    REPORT_DIR,
    PLOT_DIR,
]:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("STEP 19 - CPU-FRIENDLY MULTILINGUAL XLM-R BASELINE")
print("=" * 70)

print()
print("EXPERIMENT PURPOSE")
print("-" * 70)
print("Testing XLM-R across Tamil, Malayalam and Kannada.")
print("This is the MULTILINGUAL BASELINE.")
print()
print("No class weighting.")
print("No oversampling.")
print("No focal loss.")
print("No adaptive loss.")
print()


# ============================================================
# DEVICE
# ============================================================

print("=" * 70)
print("1. COMPUTING DEVICE")
print("=" * 70)

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("GPU available.")
    print("Device:", torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("GPU not available.")
    print("Using CPU-friendly training.")
    print("Device: CPU")


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("2. LOADING DRAVIDIAN DATASET")
print("=" * 70)

print()
print("Reading:")
print(DATA_PATH)

if not os.path.exists(DATA_PATH):
    print()
    print("ERROR: Dataset not found.")
    print(DATA_PATH)
    sys.exit(1)

df = pd.read_csv(DATA_PATH)

print()
print("Dataset loaded successfully.")
print("Total records:", len(df))
print("Columns:", list(df.columns))


required_columns = [
    "text",
    "mapped_label",
    "language",
    "split",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    print()
    print("ERROR: Missing columns:")
    print(missing)
    sys.exit(1)


# ============================================================
# CLEAN TEXT
# ============================================================

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[df["text"] != ""].copy()


# ============================================================
# LABEL MAPPING
# ============================================================

print()
print("=" * 70)
print("3. LABEL MAPPING")
print("=" * 70)

label_mapping = {
    "Non_Offensive": 0,
    "Offensive": 1,
}

df["labels"] = df["mapped_label"].map(label_mapping)

if df["labels"].isna().any():
    print()
    print("ERROR: Unknown labels detected.")
    print(df.loc[df["labels"].isna(), "mapped_label"].unique())
    sys.exit(1)

df["labels"] = df["labels"].astype(int)

id_to_label = {
    0: "Non_Offensive",
    1: "Offensive",
}

print()
print("Classes:")
print("0 -> Non_Offensive")
print("1 -> Offensive")


# ============================================================
# ORIGINAL SPLITS
# ============================================================

print()
print("=" * 70)
print("4. USING ORIGINAL DATA SPLITS")
print("=" * 70)

train_df = df[df["split"] == "train"].copy()
dev_df = df[df["split"] == "dev"].copy()
test_df = df[df["split"] == "test"].copy()

print()
print("Original records:")
print("Training  :", len(train_df))
print("Validation:", len(dev_df))
print("Testing   :", len(test_df))


# ============================================================
# STRATIFIED MULTILINGUAL TRAINING SUBSET
# ============================================================

print()
print("=" * 70)
print("5. CREATING CPU-FRIENDLY MULTILINGUAL TRAINING SUBSET")
print("=" * 70)

print()
print("Requested training samples :", TRAINING_SAMPLES)
print("Requested validation samples:", VALIDATION_SAMPLES)

print()
print("Important:")
print("The COMPLETE test set will be preserved.")
print("Only training/validation are reduced for CPU feasibility.")


def stratified_multilingual_sample(
    data,
    target_size,
    seed=42
):
    """
    Sample proportionally across language + label combinations.
    This keeps both multilingual and class structure.
    """

    if target_size >= len(data):
        return data.copy()

    grouped = data.groupby(
        ["language", "labels"],
        group_keys=False
    )

    parts = []

    total = len(data)

    for (_, _), group in grouped:

        proportion = len(group) / total

        n = max(
            1,
            int(round(target_size * proportion))
        )

        n = min(n, len(group))

        parts.append(
            group.sample(
                n=n,
                random_state=seed
            )
        )

    sampled = pd.concat(
        parts,
        ignore_index=True
    )

    # Correct possible rounding overflow
    if len(sampled) > target_size:

        sampled = sampled.sample(
            n=target_size,
            random_state=seed
        )

    elif len(sampled) < target_size:

        remaining = data.drop(
            sampled.index,
            errors="ignore"
        )

        extra_needed = target_size - len(sampled)

        if len(remaining) > 0:

            extra = remaining.sample(
                n=min(extra_needed, len(remaining)),
                random_state=seed
            )

            sampled = pd.concat(
                [sampled, extra],
                ignore_index=True
            )

    return sampled.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)


train_small = stratified_multilingual_sample(
    train_df,
    TRAINING_SAMPLES,
    SEED
)

dev_small = stratified_multilingual_sample(
    dev_df,
    VALIDATION_SAMPLES,
    SEED
)


print()
print("Actual training samples :", len(train_small))
print("Actual validation samples:", len(dev_small))
print("Final testing samples   :", len(test_df))


# ============================================================
# SHOW LANGUAGE DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("6. MULTILINGUAL DISTRIBUTION")
print("=" * 70)

print()
print("Training language distribution:")
print(
    train_small["language"].value_counts()
)

print()
print("Validation language distribution:")
print(
    dev_small["language"].value_counts()
)

print()
print("Testing language distribution:")
print(
    test_df["language"].value_counts()
)


print()
print("Training language × label:")
print(
    pd.crosstab(
        train_small["language"],
        train_small["mapped_label"]
    )
)


# ============================================================
# SAVE EXPERIMENT DATA
# ============================================================

train_small.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "training_subset.csv"
    ),
    index=False
)

dev_small.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "validation_subset.csv"
    ),
    index=False
)


# ============================================================
# CREATE HUGGINGFACE DATASETS
# ============================================================

print()
print("=" * 70)
print("7. CREATING TRANSFORMER DATASETS")
print("=" * 70)

train_dataset = Dataset.from_pandas(
    train_small[
        ["text", "labels"]
    ],
    preserve_index=False
)

dev_dataset = Dataset.from_pandas(
    dev_small[
        ["text", "labels"]
    ],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[
        ["text", "labels"]
    ],
    preserve_index=False
)


# ============================================================
# TOKENIZER
# ============================================================

print()
print("=" * 70)
print("8. LOADING XLM-R TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_function(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )


print()
print("Tokenizing training data...")

train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    desc="Training tokenization"
)

print()
print("Tokenizing validation data...")

dev_tokenized = dev_dataset.map(
    tokenize_function,
    batched=True,
    desc="Validation tokenization"
)

print()
print("Tokenizing testing data...")

test_tokenized = test_dataset.map(
    tokenize_function,
    batched=True,
    desc="Testing tokenization"
)

print()
print("Tokenization completed.")


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 70)
print("9. LOADING XLM-R CLASSIFICATION MODEL")
print("=" * 70)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id_to_label,
    label2id=label_mapping
)

print()
print("Model loaded successfully.")


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
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
        "weighted_f1": weighted_f1,
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

print()
print("=" * 70)
print("10. CONFIGURING CPU-FRIENDLY TRAINING")
print("=" * 70)

print()
print("Training samples :", len(train_small))
print("Validation samples:", len(dev_small))
print("Testing samples  :", len(test_df))
print()
print("Epochs            :", EPOCHS)
print("Batch size        :", BATCH_SIZE)
print("Gradient accumulation:", GRADIENT_ACCUMULATION)
print("Learning rate     :", LEARNING_RATE)
print("Maximum length    :", MAX_LENGTH)
print("Checkpoint steps  :", SAVE_STEPS)
print()
print("Multilingual training is ENABLED.")
print("No class weighting is used.")
print("No oversampling is used.")
print("No adaptive loss is used.")
print("This is the MULTILINGUAL BASELINE.")


# ============================================================
# COMPATIBILITY WITH INSTALLED TRANSFORMERS
# ============================================================

training_signature = inspect.signature(
    TrainingArguments.__init__
)

trainer_signature = inspect.signature(
    Trainer.__init__
)


training_kwargs = {
    "output_dir": CHECKPOINT_DIR,
    "num_train_epochs": EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
    "learning_rate": LEARNING_RATE,
    "warmup_ratio": WARMUP_RATIO,
    "logging_steps": LOGGING_STEPS,
    "save_steps": SAVE_STEPS,
    "save_total_limit": 2,
    "load_best_model_at_end": False,
    "report_to": "none",
    "seed": SEED,
    "dataloader_pin_memory": False,
}


# ------------------------------------------------------------
# Transformers version compatibility
# ------------------------------------------------------------

if "evaluation_strategy" in training_signature.parameters:

    training_kwargs["evaluation_strategy"] = "steps"

    if "eval_steps" in training_signature.parameters:
        training_kwargs["eval_steps"] = SAVE_STEPS

elif "eval_strategy" in training_signature.parameters:

    training_kwargs["eval_strategy"] = "steps"

    if "eval_steps" in training_signature.parameters:
        training_kwargs["eval_steps"] = SAVE_STEPS


# ------------------------------------------------------------
# fp16 only if GPU exists
# ------------------------------------------------------------

if "fp16" in training_signature.parameters:

    training_kwargs["fp16"] = (
        DEVICE == "cuda"
    )


# ------------------------------------------------------------
# Remove unsupported arguments
# ------------------------------------------------------------

training_kwargs = {
    key: value
    for key, value in training_kwargs.items()
    if key in training_signature.parameters
}


training_args = TrainingArguments(
    **training_kwargs
)


# ============================================================
# TRAINER
# ============================================================

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_tokenized,
    "eval_dataset": dev_tokenized,
    "compute_metrics": compute_metrics,
}


# ------------------------------------------------------------
# Transformers compatibility:
# tokenizer vs processing_class
# ------------------------------------------------------------

if "processing_class" in trainer_signature.parameters:

    trainer_kwargs["processing_class"] = tokenizer

elif "tokenizer" in trainer_signature.parameters:

    trainer_kwargs["tokenizer"] = tokenizer


trainer = Trainer(
    **trainer_kwargs
)


# ============================================================
# FIND CHECKPOINT
# ============================================================

print()
print("=" * 70)
print("11. CHECKPOINT MANAGEMENT")
print("=" * 70)


def find_latest_checkpoint(directory):

    if not os.path.exists(directory):
        return None

    checkpoints = []

    for name in os.listdir(directory):

        path = os.path.join(
            directory,
            name
        )

        if (
            os.path.isdir(path)
            and name.startswith("checkpoint-")
        ):

            try:
                number = int(
                    name.split("-")[-1]
                )

                checkpoints.append(
                    (number, path)
                )

            except ValueError:
                pass

    if not checkpoints:
        return None

    checkpoints.sort(
        key=lambda x: x[0]
    )

    return checkpoints[-1][1]


latest_checkpoint = find_latest_checkpoint(
    CHECKPOINT_DIR
)


if latest_checkpoint:

    print()
    print("Checkpoint found:")
    print(latest_checkpoint)

    print()
    print("Training will RESUME from this checkpoint.")

else:

    print()
    print("No checkpoint found.")

    print()
    print("Training will start from the beginning.")


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("12. MULTILINGUAL XLM-R TRAINING")
print("=" * 70)

print()
print("IMPORTANT:")
print("If training stops, simply run the script again.")
print("It will automatically resume from the latest checkpoint.")
print()
print("Training started...")


trainer.train(
    resume_from_checkpoint=latest_checkpoint
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print()
print("=" * 70)
print("13. SAVING FINAL MODEL")
print("=" * 70)

trainer.save_model(
    MODEL_DIR
)

tokenizer.save_pretrained(
    MODEL_DIR
)

print()
print("Final model saved:")
print(MODEL_DIR)


# ============================================================
# EVALUATE COMPLETE TEST SET
# ============================================================

print()
print("=" * 70)
print("14. FINAL TEST EVALUATION")
print("=" * 70)

test_output = trainer.predict(
    test_tokenized
)

test_predictions = np.argmax(
    test_output.predictions,
    axis=-1
)

test_labels = (
    test_output.label_ids
)


# ============================================================
# OVERALL METRICS
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


print()
print("=" * 70)
print("MULTILINGUAL XLM-R BASELINE RESULTS")
print("=" * 70)

print()
print(f"Accuracy          : {accuracy:.4f}")
print(f"Macro Precision   : {macro_precision:.4f}")
print(f"Macro Recall      : {macro_recall:.4f}")
print(f"Macro F1          : {macro_f1:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall   : {weighted_recall:.4f}")
print(f"Weighted F1       : {weighted_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("15. OVERALL CLASSIFICATION REPORT")
print("=" * 70)

report_dict = classification_report(
    test_labels,
    test_predictions,
    target_names=[
        "Non_Offensive",
        "Offensive"
    ],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report_dict
).transpose()

print()
print(report_df)


report_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "overall_classification_report.csv"
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_Non_Offensive",
        "Actual_Offensive"
    ],
    columns=[
        "Predicted_Non_Offensive",
        "Predicted_Offensive"
    ]
)

cm_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "confusion_matrix.csv"
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = test_df.copy()

predictions_df[
    "true_label_id"
] = test_labels

predictions_df[
    "predicted_label_id"
] = test_predictions

predictions_df[
    "true_label"
] = predictions_df[
    "true_label_id"
].map(id_to_label)

predictions_df[
    "predicted_label"
] = predictions_df[
    "predicted_label_id"
].map(id_to_label)

predictions_df[
    "correct"
] = (
    predictions_df["true_label_id"]
    ==
    predictions_df["predicted_label_id"]
)


predictions_path = os.path.join(
    PREDICTION_DIR,
    "multilingual_xlmr_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

print()
print("Predictions saved:")
print(predictions_path)


# ============================================================
# LANGUAGE-WISE EVALUATION
# ============================================================

print()
print("=" * 70)
print("16. LANGUAGE-WISE PERFORMANCE")
print("=" * 70)


language_results = []


for language in sorted(
    predictions_df["language"].unique()
):

    language_df = predictions_df[
        predictions_df["language"] == language
    ]

    y_true = language_df[
        "true_label_id"
    ]

    y_pred = language_df[
        "predicted_label_id"
    ]

    lang_accuracy = accuracy_score(
        y_true,
        y_pred
    )

    lang_precision, lang_recall, lang_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    language_results.append({
        "Language": language,
        "Samples": len(language_df),
        "Accuracy": lang_accuracy,
        "Macro_Precision": lang_precision,
        "Macro_Recall": lang_recall,
        "Macro_F1": lang_f1,
    })

language_results_df = pd.DataFrame(
    language_results
)

print()
print(language_results_df)


language_results_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "language_wise_performance.csv"
    ),
    index=False
)


# ============================================================
# LANGUAGE × CLASS REPORT
# ============================================================

print()
print("=" * 70)
print("17. LANGUAGE × CLASS ANALYSIS")
print("=" * 70)


language_class_results = []


for language in sorted(
    predictions_df["language"].unique()
):

    language_df = predictions_df[
        predictions_df["language"] == language
    ]

    report = classification_report(
        language_df["true_label_id"],
        language_df["predicted_label_id"],
        labels=[0, 1],
        target_names=[
            "Non_Offensive",
            "Offensive"
        ],
        output_dict=True,
        zero_division=0
    )

    for label_name in [
        "Non_Offensive",
        "Offensive"
    ]:

        language_class_results.append({
            "Language": language,
            "Class": label_name,
            "Precision": report[
                label_name
            ]["precision"],
            "Recall": report[
                label_name
            ]["recall"],
            "F1": report[
                label_name
            ]["f1-score"],
            "Support": report[
                label_name
            ]["support"],
        })


language_class_df = pd.DataFrame(
    language_class_results
)

print()
print(language_class_df)


language_class_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "language_class_performance.csv"
    ),
    index=False
)


# ============================================================
# ERROR ANALYSIS
# ============================================================

print()
print("=" * 70)
print("18. MULTILINGUAL ERROR ANALYSIS")
print("=" * 70)


errors_df = predictions_df[
    predictions_df["correct"] == False
].copy()


print()
print("Total test samples :", len(predictions_df))
print("Correct             :", predictions_df["correct"].sum())
print("Incorrect           :", len(errors_df))
print("Error rate          :", 1 - accuracy)


errors_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "multilingual_misclassified.csv"
    ),
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame([{
    "Model": "XLM-R Multilingual Baseline",
    "Training_Samples": len(train_small),
    "Validation_Samples": len(dev_small),
    "Testing_Samples": len(test_df),
    "Accuracy": accuracy,
    "Macro_Precision": macro_precision,
    "Macro_Recall": macro_recall,
    "Macro_F1": macro_f1,
    "Weighted_Precision": weighted_precision,
    "Weighted_Recall": weighted_recall,
    "Weighted_F1": weighted_f1,
    "Epochs": EPOCHS,
    "Batch_Size": BATCH_SIZE,
    "Max_Length": MAX_LENGTH,
    "Learning_Rate": LEARNING_RATE,
}])

summary_path = os.path.join(
    OUTPUT_DIR,
    "multilingual_xlmr_baseline_results.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("STEP 19 - MULTILINGUAL XLM-R BASELINE COMPLETED")
print("=" * 70)

print()
print("Final test samples:", len(test_df))

print()
print("Overall Results:")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Macro F1     : {macro_f1:.4f}")
print(f"Weighted F1  : {weighted_f1:.4f}")

print()
print("Results directory:")
print(OUTPUT_DIR)

print()
print("Generated files:")
print("- multilingual_xlmr_baseline_results.csv")
print("- training_subset.csv")
print("- validation_subset.csv")
print("- final_model/")
print("- checkpoints/")
print("- predictions/multilingual_xlmr_predictions.csv")
print("- reports/overall_classification_report.csv")
print("- reports/confusion_matrix.csv")
print("- reports/language_wise_performance.csv")
print("- reports/language_class_performance.csv")
print("- reports/multilingual_misclassified.csv")

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)
print(
    "This experiment is a MULTILINGUAL BASELINE. "
    "It does not use class weighting, oversampling, "
    "or adaptive minority-aware loss."
)

print()
print("If training is interrupted:")
print("Run this SAME script again.")
print("It will automatically resume from the latest checkpoint.")

print()
print("DO NOT delete the checkpoints directory.")

print()
print("Step 19 finished.")
print("=" * 70)