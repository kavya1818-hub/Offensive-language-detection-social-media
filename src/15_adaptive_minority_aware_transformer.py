# ============================================================
# STEP 15 - ADAPTIVE MINORITY-AWARE XLM-R TRANSFORMER
# ============================================================
#
# Research purpose:
# ------------------------------------------------------------
# Dataset 1:
#
# Step 8  -> XLM-R Baseline
# Step 10 -> Class-Weighted XLM-R
# Step 11 -> Balanced XLM-R
#
# Step 15 -> Proposed Adaptive Minority-Aware XLM-R
#
# Motivation:
# ------------------------------------------------------------
# Baseline XLM-R:
#     Hate Speech Recall = 0%
#
# Weighted XLM-R:
#     Hate Speech Recall = 0%
#     Model collapsed toward Offensive Language
#
# Balanced XLM-R:
#     Hate Speech Recall = 59.21%
#     But many Offensive -> Hate Speech false positives
#
# Proposed idea:
#     Use tempered class weights + focal loss.
#
# The objective is NOT simply maximizing accuracy.
#
# Main research targets:
#     1. Hate Speech Recall
#     2. Hate Speech F1
#     3. Macro F1
#     4. Reduction of false Hate Speech predictions
# ============================================================


import os
import time
import glob
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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "xlm-roberta-base"

TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"

OUTPUT_DIR = "models/xlm_roberta_adaptive"
CHECKPOINT_DIR = "models/xlm_roberta_adaptive_checkpoints"

RESULTS_DIR = "results/adaptive_transformer"

MAX_LENGTH = 64

# Keep the same experimental scale used in the previous
# transformer experiments for fair comparison.
TRAINING_SAMPLES = 4000
VALIDATION_SAMPLES = 1000

BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5

SEED = 42

# ------------------------------------------------------------
# Proposed loss parameters
# ------------------------------------------------------------

# Focal-loss focusing parameter.
# Higher value focuses more strongly on difficult examples.
FOCAL_GAMMA = 2.0

# Tempering parameter.
#
# Instead of using the full inverse-frequency weight:
#
#     weight = 1 / frequency
#
# we use:
#
#     weight = 1 / sqrt(frequency)
#
# and normalize the resulting weights.
#
# This reduces the extreme weighting problem seen in Step 10.
TEMPERING_POWER = 0.5


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

print("=" * 70)
print("STEP 15 - ADAPTIVE MINORITY-AWARE XLM-R TRANSFORMER")
print("=" * 70)

print("\nCOMPUTING DEVICE")
print("-" * 70)

if torch.cuda.is_available():

    DEVICE = "cuda"

    print("GPU available.")
    print("Device:", torch.cuda.get_device_name(0))

else:

    DEVICE = "cpu"

    print("GPU not available.")
    print("Using CPU-optimized adaptive transformer experiment.")
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

print("\n" + "=" * 70)
print("1. LOADING DATA")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nTraining records :", len(train_df))
print("Testing records  :", len(test_df))


# ============================================================
# DETECT TEXT COLUMN
# ============================================================

if "model_text" in train_df.columns:

    TEXT_COLUMN = "model_text"

elif "normalized_text" in train_df.columns:

    TEXT_COLUMN = "normalized_text"

elif "tweet" in train_df.columns:

    TEXT_COLUMN = "tweet"

else:

    raise ValueError(
        "No valid text column found. "
        "Expected model_text, normalized_text or tweet."
    )


# ============================================================
# DETECT LABEL COLUMN
# ============================================================

if "label" in train_df.columns:

    LABEL_COLUMN = "label"

else:

    raise ValueError(
        "Label column not found."
    )


print("\nText column :", TEXT_COLUMN)
print("Label column:", LABEL_COLUMN)


# ============================================================
# CLASS MAPPING
# ============================================================

classes = sorted(
    train_df[LABEL_COLUMN].dropna().unique()
)

label2id = {
    label: idx
    for idx, label in enumerate(classes)
}

id2label = {
    idx: label
    for label, idx in label2id.items()
}

num_classes = len(classes)

print("\nNumber of classes:", num_classes)

print("\nClasses:")

for idx, label in id2label.items():

    print(
        f"{idx} -> {label}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

train_df = train_df.dropna(
    subset=[TEXT_COLUMN, LABEL_COLUMN]
).copy()

test_df = test_df.dropna(
    subset=[TEXT_COLUMN, LABEL_COLUMN]
).copy()

train_df[TEXT_COLUMN] = (
    train_df[TEXT_COLUMN]
    .astype(str)
)

test_df[TEXT_COLUMN] = (
    test_df[TEXT_COLUMN]
    .astype(str)
)


train_df["labels"] = (
    train_df[LABEL_COLUMN]
    .map(label2id)
)

test_df["labels"] = (
    test_df[LABEL_COLUMN]
    .map(label2id)
)


# ============================================================
# ORIGINAL TEST SET
# ============================================================

final_test_df = test_df.copy()


# ============================================================
# CREATE STRATIFIED TRANSFORMER DATA
# ============================================================

print("\n" + "=" * 70)
print("2. CREATING STRATIFIED TRANSFORMER DATA")
print("=" * 70)

print(
    "\nFull training dataset:",
    len(train_df)
)


# ------------------------------------------------------------
# First create the training subset.
# ------------------------------------------------------------

train_parts = []

remaining_parts = []


for label in classes:

    class_data = train_df[
        train_df[LABEL_COLUMN] == label
    ].copy()

    proportion = (
        len(class_data)
        / len(train_df)
    )

    sample_size = int(
        round(
            TRAINING_SAMPLES
            * proportion
        )
    )

    sample_size = max(
        1,
        min(
            sample_size,
            len(class_data)
        )
    )

    sampled = class_data.sample(
        n=sample_size,
        random_state=SEED
    )

    train_parts.append(sampled)


transformer_data = pd.concat(
    train_parts,
    ignore_index=True
)


# ------------------------------------------------------------
# Correct validation split
# ------------------------------------------------------------
#
# IMPORTANT:
# We split using actual dataframe rows.
#
# This avoids accidentally deleting the first N rows,
# which can happen when validation indices are reset.
# ------------------------------------------------------------

validation_parts = []
training_parts = []


for label in classes:

    class_data = transformer_data[
        transformer_data[LABEL_COLUMN] == label
    ].copy()

    class_data = class_data.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    proportion = (
        len(class_data)
        / len(transformer_data)
    )

    validation_size = int(
        round(
            VALIDATION_SAMPLES
            * proportion
        )
    )

    validation_size = max(
        1,
        min(
            validation_size,
            len(class_data) - 1
        )
    )

    validation_part = class_data.iloc[
        :validation_size
    ].copy()

    training_part = class_data.iloc[
        validation_size:
    ].copy()

    validation_parts.append(
        validation_part
    )

    training_parts.append(
        training_part
    )


validation_df = pd.concat(
    validation_parts,
    ignore_index=True
)

transformer_train = pd.concat(
    training_parts,
    ignore_index=True
)


# Shuffle

validation_df = validation_df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

transformer_train = transformer_train.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


print(
    "\nTransformer training:",
    len(transformer_train)
)

print(
    "Validation           :",
    len(validation_df)
)

print(
    "Final testing        :",
    len(final_test_df)
)


# ============================================================
# DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("3. DATA DISTRIBUTION")
print("=" * 70)

print("\nAdaptive Training Distribution:")

print(
    transformer_train[
        LABEL_COLUMN
    ].value_counts()
)


print("\nValidation Distribution:")

print(
    validation_df[
        LABEL_COLUMN
    ].value_counts()
)


print("\nTesting Distribution:")

print(
    final_test_df[
        LABEL_COLUMN
    ].value_counts()
)


# ============================================================
# ADAPTIVE / TEMPERED CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("4. CALCULATING TEMPERED CLASS WEIGHTS")
print("=" * 70)

class_counts = (
    transformer_train["labels"]
    .value_counts()
    .sort_index()
)


print("\nClass counts:")

for idx, count in enumerate(
    class_counts.values
):

    print(
        f"{id2label[idx]:25s}: {count}"
    )


# ------------------------------------------------------------
# Tempered inverse-frequency weighting
#
# weight_i = 1 / count_i ^ power
#
# power = 0.5
#
# This is deliberately less aggressive than full
# inverse-frequency weighting.
# ------------------------------------------------------------

raw_weights = (
    1.0
    / (
        class_counts.values
        ** TEMPERING_POWER
    )
)


# Normalize mean weight to 1

raw_weights = (
    raw_weights
    / raw_weights.mean()
)


class_weights = torch.tensor(
    raw_weights,
    dtype=torch.float
)


print(
    "\nTempering power:",
    TEMPERING_POWER
)

print("\nAdaptive class weights:")

for idx, weight in enumerate(
    class_weights
):

    print(
        f"{id2label[idx]:25s}: "
        f"{weight.item():.4f}"
    )


# ============================================================
# CONVERT TO HUGGINGFACE DATASETS
# ============================================================

print("\n" + "=" * 70)
print("5. CREATING TRANSFORMER DATASETS")
print("=" * 70)


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

print("\n" + "=" * 70)
print("6. LOADING TOKENIZER")
print("=" * 70)

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
# FORMAT DATASETS
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
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("7. LOADING XLM-R")
print("=" * 70)


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

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {

        "accuracy": accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "weighted_precision":
            weighted_precision,

        "weighted_recall":
            weighted_recall,

        "weighted_f1":
            weighted_f1
    }


# ============================================================
# ADAPTIVE FOCAL LOSS
# ============================================================

class AdaptiveFocalTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.pop(
            "labels"
        )

        outputs = model(
            **inputs
        )

        logits = outputs.logits

        weights = class_weights.to(
            logits.device
        )

        # ----------------------------------------------------
        # Cross entropy per example
        # ----------------------------------------------------

        ce_loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            weight=weights,
            reduction="none"
        )

        # ----------------------------------------------------
        # Calculate probability of correct class
        # ----------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        true_class_probability = (
            probabilities
            .gather(
                1,
                labels.unsqueeze(1)
            )
            .squeeze(1)
        )

        # ----------------------------------------------------
        # Focal factor
        #
        # Easy examples:
        #     high probability
        #     lower contribution
        #
        # Difficult examples:
        #     low probability
        #     higher contribution
        # ----------------------------------------------------

        focal_factor = (
            1.0
            - true_class_probability
        ) ** FOCAL_GAMMA


        focal_loss = (
            focal_factor
            * ce_loss
        )


        loss = focal_loss.mean()


        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

print("\n" + "=" * 70)
print("8. CONFIGURING ADAPTIVE TRAINING")
print("=" * 70)


training_args = TrainingArguments(

    output_dir=CHECKPOINT_DIR,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    weight_decay=0.01,

    logging_steps=50,

    # --------------------------------------------------------
    # Checkpoint every 100 steps
    # --------------------------------------------------------

    save_strategy="steps",

    save_steps=100,

    save_total_limit=3,

    # --------------------------------------------------------
    # Evaluate at the end of training
    # --------------------------------------------------------

    eval_strategy="epoch",

    report_to="none",

    fp16=False,

    dataloader_pin_memory=False,

    seed=SEED
)


# ============================================================
# TRAINER
# ============================================================

trainer = AdaptiveFocalTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    compute_metrics=compute_metrics
)


# ============================================================
# FIND LATEST CHECKPOINT
# ============================================================

def get_latest_checkpoint():

    checkpoints = glob.glob(
        os.path.join(
            CHECKPOINT_DIR,
            "checkpoint-*"
        )
    )

    if not checkpoints:

        return None

    def checkpoint_number(path):

        name = os.path.basename(
            path
        )

        try:

            return int(
                name.split("-")[-1]
            )

        except:

            return -1


    checkpoints.sort(
        key=checkpoint_number
    )

    return checkpoints[-1]


latest_checkpoint = (
    get_latest_checkpoint()
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("9. ADAPTIVE MINORITY-AWARE XLM-R TRAINING")
print("=" * 70)


print(
    "\nTraining samples :",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(validation_dataset)
)

print(
    "Epochs            :",
    EPOCHS
)

print(
    "Maximum length    :",
    MAX_LENGTH
)

print(
    "Batch size        :",
    BATCH_SIZE
)

print(
    "Learning rate     :",
    LEARNING_RATE
)

print(
    "Focal gamma       :",
    FOCAL_GAMMA
)

print(
    "Weight power      :",
    TEMPERING_POWER
)

print(
    "\nAdaptive minority-aware loss is ENABLED."
)

if latest_checkpoint:

    print(
        "\nExisting checkpoint detected:"
    )

    print(
        latest_checkpoint
    )

    print(
        "\nTraining will resume from "
        "the latest checkpoint."
    )

else:

    print(
        "\nNo checkpoint found."
    )

    print(
        "Training starts from the beginning."
    )


print("\nTraining started...\n")


start_time = time.time()


# ------------------------------------------------------------
# RESUME IF CHECKPOINT EXISTS
# ------------------------------------------------------------

if latest_checkpoint:

    trainer.train(
        resume_from_checkpoint=latest_checkpoint
    )

else:

    trainer.train()


training_time = (
    time.time()
    - start_time
)


print("\nTraining completed.")

print(
    f"Training time: "
    f"{training_time / 60:.2f} minutes"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("10. SAVING ADAPTIVE XLM-R MODEL")
print("=" * 70)


trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


print("\nModel saved:")

print(
    OUTPUT_DIR
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("11. VALIDATION EVALUATION")
print("=" * 70)


validation_results = trainer.evaluate(
    validation_dataset
)


for key, value in validation_results.items():

    if isinstance(
        value,
        float
    ):

        print(
            f"{key:25s}: "
            f"{value:.4f}"
        )

    else:

        print(
            f"{key:25s}: "
            f"{value}"
        )


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("12. FINAL TEST EVALUATION")
print("=" * 70)


test_results = trainer.evaluate(
    test_dataset
)


print("\nADAPTIVE XLM-R TEST RESULTS")

for key, value in test_results.items():

    if key.startswith("eval_"):

        metric_name = key.replace(
            "eval_",
            ""
        )

        if isinstance(
            value,
            float
        ):

            print(
                f"{metric_name:20s}: "
                f"{value:.4f}"
            )


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("13. GENERATING TEST PREDICTIONS")
print("=" * 70)


prediction_output = trainer.predict(
    test_dataset
)


logits = prediction_output.predictions


predicted_ids = np.argmax(
    logits,
    axis=1
)


true_ids = (
    final_test_df["labels"]
    .astype(int)
    .values
)


predicted_labels = [
    id2label[int(x)]
    for x in predicted_ids
]


true_labels = [
    id2label[int(x)]
    for x in true_ids
]


# ============================================================
# CONFIDENCE
# ============================================================

probabilities = torch.softmax(
    torch.tensor(logits),
    dim=1
).numpy()


confidence = (
    probabilities.max(
        axis=1
    )
)


correct = (
    predicted_ids
    == true_ids
)


# ============================================================
# PREDICTION DATAFRAME
# ============================================================

prediction_df = final_test_df.copy()

prediction_df[
    "actual_label"
] = true_labels

prediction_df[
    "predicted_label"
] = predicted_labels

prediction_df[
    "confidence"
] = confidence

prediction_df[
    "correct"
] = correct


prediction_file = os.path.join(
    RESULTS_DIR,
    "adaptive_xlm_roberta_predictions.csv"
)


prediction_df.to_csv(
    prediction_file,
    index=False
)


print(
    "\nPrediction results saved:"
)

print(
    prediction_file
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("14. CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(
    true_labels,
    predicted_labels,
    labels=classes,
    output_dict=True,
    zero_division=0
)


report_df = pd.DataFrame(
    report
).transpose()


print(
    report_df
)


report_file = os.path.join(
    RESULTS_DIR,
    "adaptive_xlm_roberta_classification_report.csv"
)


report_df.to_csv(
    report_file
)


print(
    "\nClassification report saved:"
)

print(
    report_file
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("15. CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=classes
)


cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual_{x}"
        for x in classes
    ],
    columns=[
        f"Predicted_{x}"
        for x in classes
    ]
)


print(
    cm_df
)


cm_file = os.path.join(
    RESULTS_DIR,
    "adaptive_xlm_roberta_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_file
)


print(
    "\nConfusion matrix saved:"
)

print(
    cm_file
)


# ============================================================
# MISCLASSIFIED SAMPLES
# ============================================================

misclassified_df = (
    prediction_df[
        prediction_df["correct"] == False
    ]
    .copy()
)


misclassified_file = os.path.join(
    RESULTS_DIR,
    "adaptive_xlm_roberta_misclassified.csv"
)


misclassified_df.to_csv(
    misclassified_file,
    index=False
)


print(
    "\nMisclassified samples:",
    len(misclassified_df)
)


print(
    "Misclassified samples saved:"
)

print(
    misclassified_file
)


# ============================================================
# CLASS-WISE PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("16. CLASS-WISE PERFORMANCE")
print("=" * 70)


class_rows = []


for label in classes:

    actual_mask = (
        np.array(true_labels)
        == label
    )

    predicted_mask = (
        np.array(predicted_labels)
        == label
    )

    actual_count = (
        actual_mask.sum()
    )

    predicted_count = (
        predicted_mask.sum()
    )

    true_positive = (
        actual_mask
        & predicted_mask
    ).sum()

    precision = (
        true_positive
        / predicted_count
        if predicted_count > 0
        else 0.0
    )

    recall = (
        true_positive
        / actual_count
        if actual_count > 0
        else 0.0
    )

    if (
        precision + recall
        > 0
    ):

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0


    class_rows.append({

        "Model":
            "XLM-R Adaptive",

        "Class":
            label,

        "Actual":
            actual_count,

        "Predicted":
            predicted_count,

        "Correct":
            true_positive,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "Error_Count":
            actual_count
            - true_positive
    })


class_performance_df = pd.DataFrame(
    class_rows
)


print(
    class_performance_df
)


class_file = os.path.join(
    RESULTS_DIR,
    "adaptive_xlm_roberta_class_wise_performance.csv"
)


class_performance_df.to_csv(
    class_file,
    index=False
)


print(
    "\nClass-wise results saved:"
)

print(
    class_file
)


# ============================================================
# ERROR PAIRS
# ============================================================

print("\n" + "=" * 70)
print("17. ERROR PAIR ANALYSIS")
print("=" * 70)


error_df = prediction_df[
    prediction_df["correct"] == False
].copy()


if len(error_df) > 0:

    error_pairs = (
        error_df
        .groupby(
            [
                "actual_label",
                "predicted_label"
            ]
        )
        .size()
        .reset_index(
            name="Count"
        )
        .sort_values(
            "Count",
            ascending=False
        )
    )


    error_pairs[
        "Error_Type"
    ] = (
        error_pairs[
            "actual_label"
        ]
        + " -> "
        + error_pairs[
            "predicted_label"
        ]
    )


    error_pairs = error_pairs[
        [
            "Error_Type",
            "Count"
        ]
    ]

else:

    error_pairs = pd.DataFrame(
        columns=[
            "Error_Type",
            "Count"
        ]
    )


print(
    error_pairs
)


error_pairs_file = os.path.join(
    RESULTS_DIR,
    "adaptive_xlm_roberta_error_pairs.csv"
)


error_pairs.to_csv(
    error_pairs_file,
    index=False
)


print(
    "\nError-pair analysis saved:"
)

print(
    error_pairs_file
)


# ============================================================
# FINAL RESULTS CSV
# ============================================================

print("\n" + "=" * 70)
print("18. SAVING FINAL RESULTS")
print("=" * 70)


def get_metric(
    results,
    name
):

    key = (
        "eval_"
        + name
    )

    return float(
        results.get(
            key,
            0.0
        )
    )


final_results = {

    "Model":
        "XLM-R Adaptive Minority-Aware",

    "Accuracy":
        get_metric(
            test_results,
            "accuracy"
        ),

    "Macro_Precision":
        get_metric(
            test_results,
            "macro_precision"
        ),

    "Macro_Recall":
        get_metric(
            test_results,
            "macro_recall"
        ),

    "Macro_F1":
        get_metric(
            test_results,
            "macro_f1"
        ),

    "Weighted_Precision":
        get_metric(
            test_results,
            "weighted_precision"
        ),

    "Weighted_Recall":
        get_metric(
            test_results,
            "weighted_recall"
        ),

    "Weighted_F1":
        get_metric(
            test_results,
            "weighted_f1"
        ),

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
        EPOCHS,

    "Batch_Size":
        BATCH_SIZE,

    "Focal_Gamma":
        FOCAL_GAMMA,

    "Weight_Tempering_Power":
        TEMPERING_POWER
}


results_df = pd.DataFrame(
    [final_results]
)


results_file = os.path.join(
    RESULTS_DIR,
    "adaptive_xlm_roberta_results.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


print(
    "\nFinal results saved:"
)

print(
    results_file
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 15 - ADAPTIVE TRANSFORMER COMPLETED")
print("=" * 70)


print("\nFINAL ADAPTIVE XLM-R RESULTS:")

print(
    f"Accuracy   : "
    f"{final_results['Accuracy']:.4f}"
)

print(
    f"Macro F1   : "
    f"{final_results['Macro_F1']:.4f}"
)

print(
    f"Weighted F1: "
    f"{final_results['Weighted_F1']:.4f}"
)


# Find Hate Speech performance

hate_label = "Hate_Speech"


hate_row = class_performance_df[
    class_performance_df["Class"]
    == hate_label
]


if len(hate_row) > 0:

    hate_row = hate_row.iloc[0]

    print(
        f"Hate Speech Precision: "
        f"{hate_row['Precision']:.4f}"
    )

    print(
        f"Hate Speech Recall   : "
        f"{hate_row['Recall']:.4f}"
    )

    print(
        f"Hate Speech F1       : "
        f"{hate_row['F1']:.4f}"
    )


print("\nGenerated files:")

print(
    f"- {OUTPUT_DIR}/"
)

print(
    f"- {CHECKPOINT_DIR}/"
)

print(
    f"- {RESULTS_DIR}/"
    "adaptive_xlm_roberta_results.csv"
)

print(
    f"- {RESULTS_DIR}/"
    "adaptive_xlm_roberta_classification_report.csv"
)

print(
    f"- {RESULTS_DIR}/"
    "adaptive_xlm_roberta_confusion_matrix.csv"
)

print(
    f"- {RESULTS_DIR}/"
    "adaptive_xlm_roberta_predictions.csv"
)

print(
    f"- {RESULTS_DIR}/"
    "adaptive_xlm_roberta_misclassified.csv"
)

print(
    f"- {RESULTS_DIR}/"
    "adaptive_xlm_roberta_class_wise_performance.csv"
)

print(
    f"- {RESULTS_DIR}/"
    "adaptive_xlm_roberta_error_pairs.csv"
)

print("\n" + "=" * 70)
print("STEP 15 FINISHED")
print("=" * 70)