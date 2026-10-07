import os
import sys
import random
import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

DATASET1_PATH = os.path.join(
    DATA_DIR,
    "train.csv"
)

DATASET2_PATH = os.path.join(
    DATA_DIR,
    "dravidian_clean",
    "dravidian_multilingual_clean.csv"
)

MODEL_NAME = "xlm-roberta-base"

OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "results",
    "multilingual_cross_dataset_final"
)

CHECKPOINT_DIR = os.path.join(
    OUTPUT_ROOT,
    "checkpoints"
)

FINAL_MODEL_DIR = os.path.join(
    OUTPUT_ROOT,
    "final_model"
)

METRICS_DIR = os.path.join(
    OUTPUT_ROOT,
    "metrics"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    METRICS_DIR,
    exist_ok=True
)


# ======================================================================
# CPU CONFIGURATION
# ======================================================================

SEED = 42

MAX_LENGTH = 96

TRAIN_BATCH_SIZE = 8

EVAL_BATCH_SIZE = 8

GRADIENT_ACCUMULATION = 1

EPOCHS = 1

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

FOCAL_GAMMA = 2.0

TARGET_PER_CLASS_LANGUAGE = 1500

SAVE_STEPS = 250

LOGGING_STEPS = 50

EVAL_STEPS = 250

CPU_THREADS = 4

set_seed(SEED)

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

try:
    torch.set_num_threads(CPU_THREADS)
except Exception:
    pass


print("=" * 80)
print("COMPUTING DEVICE")
print("=" * 80)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU available.")
else:
    print("GPU not available.")
    print(
        "Using CPU-friendly training. "
        f"CPU threads: {CPU_THREADS}"
    )


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_language(value):

    value = normalize_text(value).lower()

    if "tamil" in value:
        return "Tamil"

    if "malayalam" in value:
        return "Malayalam"

    if "kannada" in value:
        return "Kannada"

    if value in [
        "english",
        "en"
    ]:
        return "English"

    return value.title()


def normalize_dravidian_label(value):

    value = normalize_text(value).lower()

    if value in [
        "non_offensive",
        "non-offensive",
        "non offensive",
        "nonoffensive",
        "not offensive",
        "not_offensive"
    ]:
        return 0

    if value in [
        "offensive",
        "offence",
        "offensive_language"
    ]:
        return 1

    return np.nan


def normalize_english_label(value):

    value = normalize_text(value)

    # Dataset 1:
    # Non_Toxic = Non_Offensive
    # Every toxic category = Offensive

    if value.lower() == "non_toxic":
        return 0

    toxic_categories = {
        "Homophobia_LGBTQphobia",
        "Misogyny_Sexism",
        "Racism_Xenophobia",
        "Harassment_Stalking",
        "Religious_Hate",
        "Profanity_Insult"
    }

    if value in toxic_categories:
        return 1

    return np.nan


def print_distribution(df, title):

    print("\n" + title)

    if len(df) == 0:
        print("Empty dataset.")
        return

    print(
        df["label_id"].value_counts().sort_index()
    )


def remove_exact_duplicates(
    df,
    text_column="text"
):

    before = len(df)

    df = df.copy()

    df[text_column] = (
        df[text_column]
        .astype(str)
        .str.strip()
    )

    df = df[
        df[text_column].str.len() > 0
    ].copy()

    df = df.drop_duplicates(
        subset=[text_column, "label_id"],
        keep="first"
    ).reset_index(drop=True)

    removed = before - len(df)

    return df, removed


def sample_class(
    df,
    language,
    label,
    target,
    seed=42
):

    subset = df[
        (df["language"] == language)
        &
        (df["label_id"] == label)
    ].copy()

    available = len(subset)

    if available == 0:
        raise ValueError(
            f"No samples available for "
            f"{language} / label {label}"
        )

    replace = available < target

    sampled = subset.sample(
        n=target,
        replace=replace,
        random_state=seed
    )

    return sampled.reset_index(drop=True), available, replace


def create_balanced_training_data(
    df1,
    df2
):

    print("=" * 80)
    print("4. CREATING LANGUAGE-BALANCED TRAINING DATA")
    print("=" * 80)

    print("""
Training strategy:

English      → balanced
Tamil        → balanced
Malayalam    → balanced using oversampling if necessary
Kannada      → balanced using oversampling if necessary

Oversampling is performed ONLY inside the training set.

Validation and test data are never oversampled.
""")

    pieces = []

    languages = [
        "English",
        "Tamil",
        "Malayalam",
        "Kannada"
    ]

    for language in languages:

        source = pd.concat(
            [
                df1[
                    df1["language"] == language
                ],
                df2[
                    df2["language"] == language
                ]
            ],
            ignore_index=True
        )

        for label in [0, 1]:

            sampled, available, replacement = (
                sample_class(
                    source,
                    language,
                    label,
                    TARGET_PER_CLASS_LANGUAGE,
                    SEED
                )
            )

            label_name = (
                "Non_Offensive"
                if label == 0
                else "Offensive"
            )

            replacement_text = (
                "WITH replacement"
                if replacement
                else "without replacement"
            )

            print(
                f"{language:<10} | "
                f"{label_name:<15} | "
                f"available={available:5d} | "
                f"used={len(sampled):4d} | "
                f"{replacement_text}"
            )

            pieces.append(sampled)

    balanced = pd.concat(
        pieces,
        ignore_index=True
    )

    balanced = balanced.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    print(
        "\nFINAL TRAINING SIZE:",
        len(balanced)
    )

    print("\nTraining distribution:")

    print(
        balanced.groupby(
            ["language", "label_id"]
        ).size()
    )

    return balanced


def create_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[1],
            average=None,
            zero_division=0
        )
    )

    return {
        "Accuracy": accuracy,
        "Macro_Precision": macro_precision,
        "Macro_Recall": macro_recall,
        "Macro_F1": macro_f1,
        "Weighted_Precision": weighted_precision,
        "Weighted_Recall": weighted_recall,
        "Weighted_F1": weighted_f1,
        "Offensive_Precision": precision[0],
        "Offensive_Recall": recall[0],
        "Offensive_F1": f1[0],
        "Offensive_Support": int(
            np.sum(np.array(y_true) == 1)
        )
    }


def compute_language_metrics(
    df,
    predictions
):

    results = []

    df = df.copy()

    df["prediction"] = predictions

    for language in [
        "English",
        "Tamil",
        "Malayalam",
        "Kannada"
    ]:

        subset = df[
            df["language"] == language
        ]

        if len(subset) == 0:
            continue

        y_true = subset[
            "label_id"
        ].values

        y_pred = subset[
            "prediction"
        ].values

        metrics = create_metrics(
            y_true,
            y_pred
        )

        row = {
            "Model":
                "Final Cross-Dataset Multilingual XLM-R",
            "Language":
                language,
            "Samples":
                len(subset)
        }

        row.update(metrics)

        results.append(row)

    return pd.DataFrame(results)


# ======================================================================
# 1. LOAD DATASET 1
# ======================================================================

print("=" * 80)
print("1. LOADING DATASET 1 - ENGLISH")
print("=" * 80)

print("\nPath:")
print(DATASET1_PATH)

if not os.path.exists(DATASET1_PATH):

    raise FileNotFoundError(
        f"Dataset 1 not found:\n{DATASET1_PATH}"
    )

df1_raw = pd.read_csv(
    DATASET1_PATH
)

print("\nRecords:", len(df1_raw))

print(
    "Columns:",
    list(df1_raw.columns)
)

if "text" not in df1_raw.columns:

    raise ValueError(
        "Dataset 1 must contain a 'text' column."
    )

if "label" not in df1_raw.columns:

    raise ValueError(
        "Dataset 1 must contain a 'label' column."
    )

print("\nText column: text")
print("Label column: label")

print("\nDATASET 1 RAW LABEL DISTRIBUTION:")

print(
    df1_raw["label"].value_counts()
)


df1 = pd.DataFrame()

df1["text"] = (
    df1_raw["text"]
    .map(normalize_text)
)

df1["label_clean"] = (
    df1_raw["label"]
    .map(normalize_text)
)

df1["label_id"] = (
    df1["label_clean"]
    .map(normalize_english_label)
)

df1["language"] = "English"


df1 = df1[
    df1["text"].str.len() > 0
].copy()


df1 = df1[
    df1["label_id"].notna()
].copy()


df1["label_id"] = (
    df1["label_id"]
    .astype(int)
)


df1, removed1 = remove_exact_duplicates(
    df1
)

print(
    "\nDataset 1 exact duplicate rows removed:",
    removed1
)

print(
    "Dataset 1 cleaned records:",
    len(df1)
)

print("\nDataset 1 mapped distribution:")

print(
    df1["label_id"].value_counts().sort_index()
)


# ======================================================================
# 2. STRATIFIED SPLIT DATASET 1
# ======================================================================

print("=" * 80)
print("2. SPLITTING DATASET 1")
print("=" * 80)

from sklearn.model_selection import train_test_split


df1_train, df1_temp = train_test_split(
    df1,
    test_size=0.20,
    stratify=df1["label_id"],
    random_state=SEED
)

df1_val, df1_test = train_test_split(
    df1_temp,
    test_size=0.50,
    stratify=df1_temp["label_id"],
    random_state=SEED
)

df1_train = df1_train.reset_index(drop=True)

df1_val = df1_val.reset_index(drop=True)

df1_test = df1_test.reset_index(drop=True)

print("\nDataset 1 split sizes:")

print(
    "Train      :",
    len(df1_train)
)

print(
    "Validation :",
    len(df1_val)
)

print(
    "Test       :",
    len(df1_test)
)

print_distribution(
    df1_train,
    "\nDataset 1 train distribution:"
)

print_distribution(
    df1_val,
    "\nDataset 1 validation distribution:"
)

print_distribution(
    df1_test,
    "\nDataset 1 test distribution:"
)


# ======================================================================
# 3. LOAD DATASET 2
# ======================================================================

print("=" * 80)
print("3. LOADING DATASET 2 - DRAVIDIAN")
print("=" * 80)

print("\nPath:")
print(DATASET2_PATH)

if not os.path.exists(DATASET2_PATH):

    raise FileNotFoundError(
        f"Dataset 2 not found:\n{DATASET2_PATH}"
    )

df2_raw = pd.read_csv(
    DATASET2_PATH
)

print("\nRecords:", len(df2_raw))

print(
    "Columns:",
    list(df2_raw.columns)
)

required2 = [
    "text",
    "mapped_label",
    "language",
    "split"
]

for column in required2:

    if column not in df2_raw.columns:

        raise ValueError(
            f"Dataset 2 missing required column: "
            f"{column}"
        )


df2 = pd.DataFrame()

df2["text"] = (
    df2_raw["text"]
    .map(normalize_text)
)

df2["label_id"] = (
    df2_raw["mapped_label"]
    .map(normalize_dravidian_label)
)

df2["language"] = (
    df2_raw["language"]
    .map(normalize_language)
)

df2["split"] = (
    df2_raw["split"]
    .map(normalize_text)
    .str.lower()
)

df2 = df2[
    df2["text"].str.len() > 0
].copy()

df2 = df2[
    df2["label_id"].notna()
].copy()

df2 = df2[
    df2["language"].isin(
        [
            "Tamil",
            "Malayalam",
            "Kannada"
        ]
    )
].copy()

df2["label_id"] = (
    df2["label_id"]
    .astype(int)
)


df2, removed2 = remove_exact_duplicates(
    df2
)

print(
    "\nDataset 2 exact duplicate rows removed:",
    removed2
)

print(
    "Dataset 2 cleaned records:",
    len(df2)
)

print("\nDataset 2 label distribution:")

print(
    df2["label_id"].value_counts().sort_index()
)

print("\nDataset 2 language distribution:")

print(
    df2["language"].value_counts()
)


# ======================================================================
# DATASET 2 SPLITS
# ======================================================================

df2_train = df2[
    df2["split"] == "train"
].copy()

df2_val = df2[
    df2["split"].isin(
        ["dev", "validation", "valid"]
    )
].copy()

df2_test = df2[
    df2["split"] == "test"
].copy()


print("\nDataset 2 split distribution:")

print(
    pd.Series(
        {
            "train": len(df2_train),
            "validation": len(df2_val),
            "test": len(df2_test)
        }
    )
)


# ======================================================================
# IMPORTANT: USE ONLY TRAINING DATA FOR BALANCING
# ======================================================================

print("\nDataset 2 original split sizes:")

print(
    "Train      :",
    len(df2_train)
)

print(
    "Validation :",
    len(df2_val)
)

print(
    "Test       :",
    len(df2_test)
)


# ======================================================================
# 4. BALANCED TRAINING DATA
# ======================================================================

train_combined_source = pd.concat(
    [
        df1_train,
        df2_train
    ],
    ignore_index=True
)

balanced_train = create_balanced_training_data(
    df1_train,
    df2_train
)


# ======================================================================
# 5. CROSS-DATASET VALIDATION
# ======================================================================

print("=" * 80)
print("5. CREATING CROSS-DATASET VALIDATION DATA")
print("=" * 80)

print("""
Validation data remains independent.

Dataset 1:
    stratified validation split

Dataset 2:
    original validation split
""")


validation_df = pd.concat(
    [
        df1_val,
        df2_val
    ],
    ignore_index=True
)

validation_df = validation_df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


print(
    "\nValidation size:",
    len(validation_df)
)

print("\nValidation distribution:")

print(
    validation_df.groupby(
        ["language", "label_id"]
    ).size()
)


# ======================================================================
# 6. INDEPENDENT TEST DATA
# ======================================================================

print("=" * 80)
print("6. CREATING INDEPENDENT TEST DATA")
print("=" * 80)

test_df = pd.concat(
    [
        df1_test,
        df2_test
    ],
    ignore_index=True
)

test_df = test_df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


print(
    "\nIndependent test size:",
    len(test_df)
)

print("\nTest distribution:")

print(
    test_df.groupby(
        ["language", "label_id"]
    ).size()
)


# ======================================================================
# 7. DATA LEAKAGE CHECK
# ======================================================================

print("=" * 80)
print("7. DATA LEAKAGE CHECK")
print("=" * 80)

train_texts = set(
    balanced_train["text"]
)

val_texts = set(
    validation_df["text"]
)

test_texts = set(
    test_df["text"]
)

train_val_overlap = len(
    train_texts.intersection(
        val_texts
    )
)

train_test_overlap = len(
    train_texts.intersection(
        test_texts
    )
)

val_test_overlap = len(
    val_texts.intersection(
        test_texts
    )
)

print(
    "Train ↔ Validation overlap:",
    train_val_overlap
)

print(
    "Train ↔ Test overlap:",
    train_test_overlap
)

print(
    "Validation ↔ Test overlap:",
    val_test_overlap
)

if (
    train_val_overlap > 0
    or
    train_test_overlap > 0
    or
    val_test_overlap > 0
):

    raise ValueError(
        "DATA LEAKAGE DETECTED. "
        "Training stopped."
    )


# ======================================================================
# 8. CREATE TRANSFORMER DATASETS
# ======================================================================

print("=" * 80)
print("8. CREATING TRANSFORMER DATASETS")
print("=" * 80)

train_hf = Dataset.from_pandas(
    balanced_train[
        [
            "text",
            "label_id"
        ]
    ],
    preserve_index=False
)

val_hf = Dataset.from_pandas(
    validation_df[
        [
            "text",
            "label_id"
        ]
    ],
    preserve_index=False
)

test_hf = Dataset.from_pandas(
    test_df[
        [
            "text",
            "label_id"
        ]
    ],
    preserve_index=False
)

print(
    "Training dataset   :",
    len(train_hf)
)

print(
    "Validation dataset :",
    len(val_hf)
)

print(
    "Testing dataset    :",
    len(test_hf)
)


# ======================================================================
# 9. TOKENIZER
# ======================================================================

print("=" * 80)
print("9. LOADING XLM-R TOKENIZER")
print("=" * 80)

print(
    "Model:",
    MODEL_NAME
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print(
    "Tokenizer loaded successfully."
)


def tokenize_function(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


print("\nTokenizing training data...")

train_hf = train_hf.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing training"
)

print("\nTokenizing validation data...")

val_hf = val_hf.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing validation"
)

print("\nTokenizing test data...")

test_hf = test_hf.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing test"
)

train_hf = train_hf.rename_column(
    "label_id",
    "labels"
)

val_hf = val_hf.rename_column(
    "label_id",
    "labels"
)

test_hf = test_hf.rename_column(
    "label_id",
    "labels"
)

columns_to_keep = [
    "input_ids",
    "attention_mask",
    "labels"
]

if "token_type_ids" in train_hf.column_names:
    columns_to_keep.append(
        "token_type_ids"
    )

train_hf.set_format(
    type="torch",
    columns=columns_to_keep
)

val_hf.set_format(
    type="torch",
    columns=columns_to_keep
)

test_hf.set_format(
    type="torch",
    columns=columns_to_keep
)

print(
    "\nTokenization completed."
)


# ======================================================================
# 10. LOAD XLM-R
# ======================================================================

print("=" * 80)
print("10. LOADING XLM-R CLASSIFICATION MODEL")
print("=" * 80)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "Non_Offensive",
        1: "Offensive"
    },
    label2id={
        "Non_Offensive": 0,
        "Offensive": 1
    }
)

print(
    "XLM-R loaded successfully."
)

print(
    "Number of labels:",
    model.config.num_labels
)


# ======================================================================
# 11. CLASS WEIGHTS
# ======================================================================

print("=" * 80)
print("11. CALCULATING MINORITY-AWARE CLASS WEIGHTS")
print("=" * 80)

class_counts = (
    balanced_train["label_id"]
    .value_counts()
    .sort_index()
)

count0 = int(
    class_counts.get(0, 0)
)

count1 = int(
    class_counts.get(1, 0)
)

total = count0 + count1

if count0 == 0 or count1 == 0:

    raise ValueError(
        "Training data does not contain both classes."
    )


# Balanced data means the weights will normally be
# close to 1.0. This is intentional.

raw_weights = np.array(
    [
        total / (2 * count0),
        total / (2 * count1)
    ],
    dtype=np.float32
)

# Minority-aware power

weights = np.power(
    raw_weights,
    0.5
)

weights = (
    weights / weights.mean()
)

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
)

print("\nTraining class counts:")

print(
    "Non_Offensive:",
    count0
)

print(
    "Offensive    :",
    count1
)

print(
    "\nMinority-aware class weights:"
)

print(
    "Non_Offensive:",
    f"{class_weights[0].item():.4f}"
)

print(
    "Offensive    :",
    f"{class_weights[1].item():.4f}"
)


# ======================================================================
# 12. FOCAL LOSS
# ======================================================================

print("=" * 80)
print("12. CONFIGURING MINORITY-AWARE FOCAL LOSS")
print("=" * 80)

print(
    "Focal gamma:",
    FOCAL_GAMMA
)

print(
    "Minority-aware focal loss ENABLED."
)


class MinorityAwareFocalTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):

        labels = inputs.pop(
            "labels"
        )

        outputs = model(
            **inputs
        )

        logits = outputs.logits

        device = logits.device

        weights = class_weights.to(
            device
        )

        ce_loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            weight=weights,
            reduction="none"
        )

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

        focal_factor = torch.pow(
            1.0 - true_class_probability,
            FOCAL_GAMMA
        )

        loss = (
            focal_factor * ce_loss
        ).mean()

        if return_outputs:

            return loss, outputs

        return loss


# ======================================================================
# 13. CPU TRAINING CONFIGURATION
# ======================================================================

print("=" * 80)
print("13. CONFIGURING CPU-FRIENDLY TRAINING")
print("=" * 80)

print(
    "Epochs                 :",
    EPOCHS
)

print(
    "Learning rate          :",
    LEARNING_RATE
)

print(
    "Training batch size    :",
    TRAIN_BATCH_SIZE
)

print(
    "Evaluation batch size  :",
    EVAL_BATCH_SIZE
)

print(
    "Gradient accumulation  :",
    GRADIENT_ACCUMULATION
)

print(
    "Maximum sequence length:",
    MAX_LENGTH
)

print(
    "Checkpoint every       :",
    SAVE_STEPS,
    "steps"
)

print(
    "Checkpoint directory   :",
    CHECKPOINT_DIR
)


# ======================================================================
# METRICS
# ======================================================================

def compute_eval_metrics(
    eval_prediction
):

    predictions, labels = (
        eval_prediction
    )

    if isinstance(
        predictions,
        tuple
    ):

        predictions = predictions[0]

    predictions = np.argmax(
        predictions,
        axis=-1
    )

    metrics = create_metrics(
        labels,
        predictions
    )

    return {
        "accuracy":
            metrics["Accuracy"],

        "macro_precision":
            metrics["Macro_Precision"],

        "macro_recall":
            metrics["Macro_Recall"],

        "macro_f1":
            metrics["Macro_F1"],

        "weighted_f1":
            metrics["Weighted_F1"],

        "offensive_precision":
            metrics["Offensive_Precision"],

        "offensive_recall":
            metrics["Offensive_Recall"],

        "offensive_f1":
            metrics["Offensive_F1"]
    }


# ======================================================================
# TRAINING ARGUMENTS - COMPATIBILITY SAFE
# ======================================================================

print("\nCreating TrainingArguments...")

base_training_args = {

    "output_dir":
        CHECKPOINT_DIR,

    "num_train_epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "per_device_train_batch_size":
        TRAIN_BATCH_SIZE,

    "per_device_eval_batch_size":
        EVAL_BATCH_SIZE,

    "gradient_accumulation_steps":
        GRADIENT_ACCUMULATION,

    "logging_steps":
        LOGGING_STEPS,

    "save_strategy":
        "steps",

    "save_steps":
        SAVE_STEPS,

    "save_total_limit":
        5,

    "report_to":
        [],

    "dataloader_num_workers":
        0,

    "seed":
        SEED,

    "weight_decay":
        WEIGHT_DECAY
}


# ----------------------------------------------------------------------
# Try modern Transformers first.
# ----------------------------------------------------------------------

try:

    training_args = TrainingArguments(
        **base_training_args,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=False,
        fp16=False,
        bf16=False,
        remove_unused_columns=True
    )

    print(
        "Modern TrainingArguments configuration accepted."
    )

except TypeError as error1:

    print(
        "\nModern TrainingArguments not supported."
    )

    print(
        "Trying older Transformers compatibility mode..."
    )

    try:

        training_args = TrainingArguments(
            **base_training_args,
            evaluation_strategy="steps",
            eval_steps=EVAL_STEPS,
            fp16=False,
            remove_unused_columns=True
        )

        print(
            "Older TrainingArguments configuration accepted."
        )

    except TypeError as error2:

        print(
            "\nTrying maximum compatibility mode..."
        )

        compatibility_args = {
            "output_dir":
                CHECKPOINT_DIR,

            "num_train_epochs":
                EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "per_device_train_batch_size":
                TRAIN_BATCH_SIZE,

            "per_device_eval_batch_size":
                EVAL_BATCH_SIZE,

            "gradient_accumulation_steps":
                GRADIENT_ACCUMULATION,

            "logging_steps":
                LOGGING_STEPS,

            "save_steps":
                SAVE_STEPS,

            "save_total_limit":
                5,

            "report_to":
                [],

            "seed":
                SEED,

            "weight_decay":
                WEIGHT_DECAY
        }

        try:

            training_args = TrainingArguments(
                **compatibility_args
            )

            print(
                "Maximum compatibility configuration accepted."
            )

        except TypeError as error3:

            print(
                "\nERROR: TrainingArguments could not "
                "be created."
            )

            print(
                "\nModern error:"
            )

            print(error1)

            print(
                "\nOlder error:"
            )

            print(error2)

            print(
                "\nCompatibility error:"
            )

            print(error3)

            print(
                "\nCheck your Transformers version with:"
            )

            print(
                'python -c "import transformers; '
                'print(transformers.__version__)"'
            )

            sys.exit(1)


# ======================================================================
# 14. INITIALIZE TRAINER - TRANSFORMERS VERSION COMPATIBILITY
# ======================================================================

print("=" * 80)
print("14. INITIALIZING TRAINER")
print("=" * 80)

import inspect

trainer_init_parameters = inspect.signature(
    MinorityAwareFocalTrainer.__init__
).parameters


trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_hf,
    "eval_dataset": val_hf,
    "compute_metrics": compute_eval_metrics
}


# ------------------------------------------------------------
# Newer Transformers versions
# ------------------------------------------------------------

if "processing_class" in trainer_init_parameters:

    trainer_kwargs["processing_class"] = tokenizer

    print(
        "Using Trainer processing_class=tokenizer"
    )


# ------------------------------------------------------------
# Older Transformers versions
# ------------------------------------------------------------

elif "tokenizer" in trainer_init_parameters:

    trainer_kwargs["tokenizer"] = tokenizer

    print(
        "Using Trainer tokenizer=tokenizer"
    )


# ------------------------------------------------------------
# Very old/custom Trainer
# ------------------------------------------------------------

else:

    print(
        "Trainer does not accept tokenizer/processing_class."
    )

    print(
        "Tokenizer will still be used during tokenization."
    )


trainer = MinorityAwareFocalTrainer(
    **trainer_kwargs
)

print(
    "Trainer initialized successfully."
)

# ======================================================================
# 15. FIND LATEST CHECKPOINT
# ======================================================================

def find_latest_checkpoint():

    if not os.path.isdir(
        CHECKPOINT_DIR
    ):

        return None

    checkpoints = []

    for name in os.listdir(
        CHECKPOINT_DIR
    ):

        path = os.path.join(
            CHECKPOINT_DIR,
            name
        )

        if not os.path.isdir(path):
            continue

        if not name.startswith(
            "checkpoint-"
        ):
            continue

        try:

            step = int(
                name.split("-")[-1]
            )

            checkpoints.append(
                (
                    step,
                    path
                )
            )

        except ValueError:

            continue

    if not checkpoints:
        return None

    checkpoints.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return checkpoints[0][1]


resume_checkpoint = (
    find_latest_checkpoint()
)


# ======================================================================
# 16. TRAIN FINAL XLM-R
# ======================================================================

print("=" * 80)
print("15. TRAINING FINAL CROSS-DATASET XLM-R")
print("=" * 80)

if resume_checkpoint:

    print(
        "\nPrevious checkpoint found:"
    )

    print(
        resume_checkpoint
    )

    print(
        "\nTraining will RESUME from this checkpoint."
    )

else:

    print(
        "\nNo previous checkpoint found."
    )

    print(
        "Training will start from xlm-roberta-base."
    )


print(
    "\nIMPORTANT:"
)

print(
    "Do not delete the checkpoint directory "
    "while training is running."
)

print(
    "\nTraining samples:",
    len(train_hf)
)

print(
    "Validation samples:",
    len(val_hf)
)

print(
    "Independent test samples:",
    len(test_hf)
)

print(
    "\nCheckpoint interval:",
    SAVE_STEPS,
    "steps"
)

print(
    "\nTraining started..."
)


try:

    if resume_checkpoint:

        train_result = trainer.train(
            resume_from_checkpoint=
                resume_checkpoint
        )

    else:

        train_result = trainer.train()

except KeyboardInterrupt:

    print(
        "\n\nTraining interrupted by user."
    )

    print(
        "Any completed checkpoint has been preserved."
    )

    print(
        "Run the same command again to resume."
    )

    sys.exit(0)

except Exception as error:

    print(
        "\n\nTRAINING ERROR:"
    )

    print(
        str(error)
    )

    print(
        "\nIf a checkpoint was already created,"
    )

    print(
        "the next run will attempt to resume from it."
    )

    raise


# ======================================================================
# 17. SAVE FINAL MODEL
# ======================================================================

print("=" * 80)
print("16. SAVING FINAL MODEL")
print("=" * 80)

os.makedirs(
    FINAL_MODEL_DIR,
    exist_ok=True
)

trainer.save_model(
    FINAL_MODEL_DIR
)

tokenizer.save_pretrained(
    FINAL_MODEL_DIR
)

print(
    "\nFinal model saved:"
)

print(
    FINAL_MODEL_DIR
)


# ======================================================================
# 18. VALIDATION EVALUATION
# ======================================================================

print("=" * 80)
print("17. FINAL VALIDATION EVALUATION")
print("=" * 80)

validation_output = (
    trainer.predict(
        val_hf
    )
)

val_logits = (
    validation_output.predictions
)

if isinstance(
    val_logits,
    tuple
):

    val_logits = val_logits[0]

val_predictions = np.argmax(
    val_logits,
    axis=-1
)

val_true = (
    validation_df[
        "label_id"
    ].values
)

val_metrics = create_metrics(
    val_true,
    val_predictions
)

print(
    "\nValidation Accuracy:",
    f"{val_metrics['Accuracy']:.4f}"
)

print(
    "Validation Macro-F1:",
    f"{val_metrics['Macro_F1']:.4f}"
)

print(
    "Validation Offensive Precision:",
    f"{val_metrics['Offensive_Precision']:.4f}"
)

print(
    "Validation Offensive Recall:",
    f"{val_metrics['Offensive_Recall']:.4f}"
)

print(
    "Validation Offensive F1:",
    f"{val_metrics['Offensive_F1']:.4f}"
)


# ======================================================================
# 19. INDEPENDENT TEST EVALUATION
# ======================================================================

print("=" * 80)
print("18. INDEPENDENT TEST EVALUATION")
print("=" * 80)

test_output = (
    trainer.predict(
        test_hf
    )
)

test_logits = (
    test_output.predictions
)

if isinstance(
    test_logits,
    tuple
):

    test_logits = test_logits[0]

test_predictions = np.argmax(
    test_logits,
    axis=-1
)

test_true = (
    test_df[
        "label_id"
    ].values
)

test_metrics = create_metrics(
    test_true,
    test_predictions
)

print(
    "\nTest Accuracy:",
    f"{test_metrics['Accuracy']:.4f}"
)

print(
    "Test Macro-F1:",
    f"{test_metrics['Macro_F1']:.4f}"
)

print(
    "Test Offensive Precision:",
    f"{test_metrics['Offensive_Precision']:.4f}"
)

print(
    "Test Offensive Recall:",
    f"{test_metrics['Offensive_Recall']:.4f}"
)

print(
    "Test Offensive F1:",
    f"{test_metrics['Offensive_F1']:.4f}"
)


# ======================================================================
# 20. OVERALL METRICS CSV
# ======================================================================

overall_row = {
    "Model":
        "Final Cross-Dataset Multilingual XLM-R",

    "Accuracy":
        test_metrics["Accuracy"],

    "Macro_Precision":
        test_metrics["Macro_Precision"],

    "Macro_Recall":
        test_metrics["Macro_Recall"],

    "Macro_F1":
        test_metrics["Macro_F1"],

    "Weighted_Precision":
        test_metrics["Weighted_Precision"],

    "Weighted_Recall":
        test_metrics["Weighted_Recall"],

    "Weighted_F1":
        test_metrics["Weighted_F1"],

    "Offensive_Precision":
        test_metrics["Offensive_Precision"],

    "Offensive_Recall":
        test_metrics["Offensive_Recall"],

    "Offensive_F1":
        test_metrics["Offensive_F1"]
}

overall_df = pd.DataFrame(
    [overall_row]
)

overall_path = os.path.join(
    METRICS_DIR,
    "overall_metrics.csv"
)

overall_df.to_csv(
    overall_path,
    index=False
)


# ======================================================================
# 21. LANGUAGE-WISE METRICS
# ======================================================================

language_metrics_df = (
    compute_language_metrics(
        test_df,
        test_predictions
    )
)

language_path = os.path.join(
    METRICS_DIR,
    "language_wise_metrics.csv"
)

language_metrics_df.to_csv(
    language_path,
    index=False
)


# ======================================================================
# 22. CONFUSION MATRIX
# ======================================================================

cm = confusion_matrix(
    test_true,
    test_predictions,
    labels=[0, 1]
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

cm_path = os.path.join(
    METRICS_DIR,
    "confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)


# ======================================================================
# 23. TEST PREDICTIONS
# ======================================================================

prediction_df = test_df.copy()

prediction_df[
    "predicted_label_id"
] = test_predictions

prediction_df[
    "predicted_label"
] = prediction_df[
    "predicted_label_id"
].map(
    {
        0: "Non_Offensive",
        1: "Offensive"
    }
)

prediction_path = os.path.join(
    METRICS_DIR,
    "test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ======================================================================
# 24. SAVE TRAINING INFORMATION
# ======================================================================

training_info = {

    "model":
        MODEL_NAME,

    "languages":
        [
            "English",
            "Tamil",
            "Malayalam",
            "Kannada"
        ],

    "architecture":
        "XLM-R contextual encoder + minority-aware focal loss",

    "dataset1":
        DATASET1_PATH,

    "dataset2":
        DATASET2_PATH,

    "training_samples":
        len(balanced_train),

    "validation_samples":
        len(validation_df),

    "test_samples":
        len(test_df),

    "max_length":
        MAX_LENGTH,

    "batch_size":
        TRAIN_BATCH_SIZE,

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "focal_gamma":
        FOCAL_GAMMA,

    "checkpoint_steps":
        SAVE_STEPS,

    "seed":
        SEED,

    "no_keyword_rules":
        True,

    "no_offensive_dictionary":
        True,

    "step20_retrained":
        True
}

info_path = os.path.join(
    OUTPUT_ROOT,
    "training_configuration.json"
)

with open(
    info_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        training_info,
        file,
        indent=4
    )


# ======================================================================
# 25. FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 80)
print("STEP 20 COMPLETED SUCCESSFULLY")
print("=" * 80)

print(
    "\nFinal multilingual model:"
)

print(
    FINAL_MODEL_DIR
)

print(
    "\nCheckpoint directory:"
)

print(
    CHECKPOINT_DIR
)

print(
    "\nMetrics directory:"
)

print(
    METRICS_DIR
)

print(
    "\nLanguages trained:"
)

print(
    "English"
)

print(
    "Tamil"
)

print(
    "Malayalam"
)

print(
    "Kannada"
)

print(
    "\nFinal Test Metrics:"
)

print(
    "Accuracy            :",
    f"{test_metrics['Accuracy']:.4f}"
)

print(
    "Macro Precision     :",
    f"{test_metrics['Macro_Precision']:.4f}"
)

print(
    "Macro Recall        :",
    f"{test_metrics['Macro_Recall']:.4f}"
)

print(
    "Macro F1            :",
    f"{test_metrics['Macro_F1']:.4f}"
)

print(
    "Offensive Precision :",
    f"{test_metrics['Offensive_Precision']:.4f}"
)

print(
    "Offensive Recall    :",
    f"{test_metrics['Offensive_Recall']:.4f}"
)

print(
    "Offensive F1        :",
    f"{test_metrics['Offensive_F1']:.4f}"
)

print(
    "\nStep 20 training finished."
)

print(
    "Step 24 should load:"
)

print(
    FINAL_MODEL_DIR
)

print("=" * 80)