# ==============================================================================
# STEP 26 - FINAL EVALUATION OF THE CURRENT STEP 24 SYSTEM
# ==============================================================================
#
# IMPORTANT:
# - Step 20 is NOT retrained.
# - Step 24 is NOT retrained.
# - This script evaluates the ALREADY TRAINED Step 24 system.
# - Dataset 1 evaluation uses ONLY the independent TEST split created with
#   the same deterministic split used in Step 24.
# - Dataset 2 evaluation uses the official TEST split after the same
#   duplicate/leakage cleaning logic used in Step 24.
# - The current Step 24 semantic classifier and calibration are loaded
#   DIRECTLY from multilingual_cross_dataset_final.
# - No keyword rules.
# - No offensive vocabulary.
# - No manual sentence rules.
#
# ==============================================================================

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score
)

warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SEED = 42

MAX_LENGTH = 96
BATCH_SIZE = 16
CPU_THREADS = min(4, os.cpu_count() or 4)

DEVICE = torch.device("cpu")

torch.set_num_threads(CPU_THREADS)

try:
    torch.set_num_interop_threads(
        max(1, min(2, CPU_THREADS))
    )
except RuntimeError:
    pass


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ------------------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------------------

DATASET1 = os.path.join(
    BASE_DIR,
    "data",
    "train.csv"
)

DATASET2 = os.path.join(
    BASE_DIR,
    "data",
    "dravidian_clean",
    "dravidian_multilingual_clean.csv"
)


# ------------------------------------------------------------------------------
# CURRENT STEP 24 ARTIFACTS
# ------------------------------------------------------------------------------

STEP24_ROOT = os.path.join(
    BASE_DIR,
    "results",
    "multilingual_cross_dataset_final"
)

STEP20_DIR = os.path.join(
    STEP24_ROOT,
    "final_model"
)

STEP24_ROOT = os.path.join(
    BASE_DIR,
    "results",
    "multilingual_cross_dataset_final",
    "step24_semantic_final"
)

SEMANTIC_CLASSIFIER_PATH = os.path.join(
    STEP24_ROOT,
    "semantic_model",
    "semantic_classifier.joblib"
)

CALIBRATION_PATH = os.path.join(
    STEP24_ROOT,
    "hybrid_calibration.json"
)

OUTPUT_DIR = os.path.join(
    STEP24_ROOT,
    "final_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


LABELS = {
    0: "Non_Offensive",
    1: "Offensive"
}

LANGUAGES = [
    "Tamil",
    "Malayalam",
    "Kannada"
]


# ==============================================================================
# PRINTING
# ==============================================================================

def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ==============================================================================
# LABEL NORMALIZATION
# ==============================================================================

def normalize_dataset1_label(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "Non_Toxic":
        return 0

    offensive_labels = {
        "Homophobia_LGBTQphobia",
        "Misogyny_Sexism",
        "Racism_Xenophobia",
        "Harassment_Stalking",
        "Religious_Hate",
        "Profanity_Insult"
    }

    if value in offensive_labels:
        return 1

    return None


def normalize_dataset2_label(value):

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {
        "0",
        "0.0",
        "non_offensive",
        "non-offensive",
        "non offensive",
        "not_offensive",
        "not offensive"
    }:
        return 0

    if value in {
        "1",
        "1.0",
        "offensive"
    }:
        return 1

    try:
        number = int(float(value))

        if number in (0, 1):
            return number

    except Exception:
        pass

    return None


# ==============================================================================
# TEXT NORMALIZATION
# ==============================================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    return " ".join(
        str(value).strip().split()
    )


def normalized_text_set(series):

    return set(
        series.astype(str)
        .str.lower()
        .str.strip()
    )


# ==============================================================================
# VERIFY CURRENT STEP 24 ARTIFACTS
# ==============================================================================

section("1. VERIFYING CURRENT STEP 24 ARTIFACTS")

required_paths = {
    "Step 20 model": STEP20_DIR,
    "Step 24 semantic classifier": SEMANTIC_CLASSIFIER_PATH,
    "Step 24 calibration": CALIBRATION_PATH
}

for name, path in required_paths.items():

    print(f"{name}:")
    print(path)

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired current Step 24 artifact was not found:\n{path}"
        )

print("\nAll current Step 24 artifacts found.")


# ==============================================================================
# LOAD CURRENT STEP 24 CALIBRATION
# ==============================================================================

section("2. LOADING CURRENT STEP 24 CALIBRATION")

with open(
    CALIBRATION_PATH,
    "r",
    encoding="utf-8"
) as file:

    calibration = json.load(file)


THRESHOLD = float(
    calibration["threshold"]
)

XLMR_WEIGHT = float(
    calibration["xlmr_weight"]
)

SEMANTIC_WEIGHT = float(
    calibration["semantic_weight"]
)

print(
    f"Decision threshold : {THRESHOLD:.4f}"
)

print(
    f"XLM-R weight       : {XLMR_WEIGHT:.4f}"
)

print(
    f"Semantic weight    : {SEMANTIC_WEIGHT:.4f}"
)

if abs(
    XLMR_WEIGHT + SEMANTIC_WEIGHT - 1.0
) > 1e-6:

    raise ValueError(
        "Step 24 calibration weights do not sum to 1."
    )


# ==============================================================================
# LOAD CURRENT STEP 20 XLM-R
# ==============================================================================

section("3. LOADING FROZEN STEP 20 XLM-R")

tokenizer = AutoTokenizer.from_pretrained(
    STEP20_DIR,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    STEP20_DIR,
    local_files_only=True
)

model.to(DEVICE)
model.eval()

print("Step 20 XLM-R loaded successfully.")
print("Step 20 is FROZEN.")
print("Step 20 will NOT be retrained.")


# ==============================================================================
# LOAD CURRENT STEP 24 SEMANTIC CLASSIFIER
# ==============================================================================

section("4. LOADING CURRENT STEP 24 SEMANTIC CLASSIFIER")

semantic_classifier = joblib.load(
    SEMANTIC_CLASSIFIER_PATH
)

print(
    "Semantic classifier loaded successfully."
)

print(
    "Step 24 semantic classifier will NOT be retrained."
)


# ==============================================================================
# XLM-R PROBABILITY
# ==============================================================================

def predict_xlmr_probability(texts):

    probabilities = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE
    ):

        batch = texts[
            start:start + BATCH_SIZE
        ]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = model(
                **encoded
            )

            probs = torch.softmax(
                outputs.logits,
                dim=-1
            )

        probabilities.extend(
            probs[:, 1]
            .cpu()
            .numpy()
            .tolist()
        )

        processed = min(
            start + len(batch),
            len(texts)
        )

        if (
            processed % 500 == 0
            or processed == len(texts)
        ):

            print(
                f"XLM-R processed "
                f"{processed}/{len(texts)}"
            )

    return np.asarray(
        probabilities,
        dtype=np.float32
    )


# ==============================================================================
# MEAN + MAX CONTEXTUAL EMBEDDINGS
# ==============================================================================

def create_embeddings(texts):

    all_embeddings = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE
    ):

        batch = texts[
            start:start + BATCH_SIZE
        ]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = model.roberta(
                **encoded
            )

            hidden = outputs.last_hidden_state

        attention_mask = (
            encoded["attention_mask"]
            .unsqueeze(-1)
            .float()
        )

        # Mean pooling
        masked_hidden = (
            hidden * attention_mask
        )

        token_count = (
            attention_mask
            .sum(dim=1)
            .clamp(min=1)
        )

        mean_pool = (
            masked_hidden.sum(dim=1)
            / token_count
        )

        # Max pooling
        max_hidden = hidden.masked_fill(
            attention_mask == 0,
            -1e9
        )

        max_pool = max_hidden.max(
            dim=1
        ).values

        # Same representation as Step 24:
        # mean + max
        representation = torch.cat(
            [
                mean_pool,
                max_pool
            ],
            dim=1
        )

        all_embeddings.append(
            representation
            .cpu()
            .numpy()
        )

        processed = min(
            start + len(batch),
            len(texts)
        )

        if (
            processed % 500 == 0
            or processed == len(texts)
        ):

            print(
                f"Embeddings processed "
                f"{processed}/{len(texts)}"
            )

    return np.vstack(
        all_embeddings
    ).astype(np.float32)


# ==============================================================================
# CURRENT STEP 24 FINAL PREDICTION
# ==============================================================================

def predict_final(dataframe):

    texts = (
        dataframe["text_clean"]
        .astype(str)
        .tolist()
    )

    print(
        "\nCreating XLM-R probabilities..."
    )

    xlmr_probability = (
        predict_xlmr_probability(texts)
    )

    print(
        "\nCreating mean + max embeddings..."
    )

    embeddings = create_embeddings(
        texts
    )

    print(
        "\nCreating semantic probabilities..."
    )

    semantic_probability = (
        semantic_classifier
        .predict_proba(embeddings)[:, 1]
    )

    hybrid_probability = (
        XLMR_WEIGHT * xlmr_probability
        +
        SEMANTIC_WEIGHT * semantic_probability
    )

    prediction = (
        hybrid_probability >= THRESHOLD
    ).astype(int)

    result = dataframe.copy()

    result["xlmr_probability"] = (
        xlmr_probability
    )

    result["semantic_probability"] = (
        semantic_probability
    )

    result["hybrid_probability"] = (
        hybrid_probability
    )

    result["predicted_label"] = (
        prediction
    )

    result["predicted_name"] = [
        LABELS[x]
        for x in prediction
    ]

    return result


# ==============================================================================
# METRICS
# ==============================================================================

def calculate_metrics(
    name,
    result
):

    y_true = (
        result["label_id"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        result["predicted_label"]
        .astype(int)
        .to_numpy()
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0
        )
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    print()
    print("-" * 80)
    print(name)
    print("-" * 80)

    print(
        f"Samples             : {len(result)}"
    )

    print(
        f"Accuracy            : {accuracy:.4f}"
    )

    print(
        f"Macro-F1            : {macro_f1:.4f}"
    )

    print()
    print("Non_Offensive")

    print(
        f"Precision           : {precision[0]:.4f}"
    )

    print(
        f"Recall              : {recall[0]:.4f}"
    )

    print(
        f"F1                  : {f1[0]:.4f}"
    )

    print()
    print("Offensive")

    print(
        f"Precision           : {precision[1]:.4f}"
    )

    print(
        f"Recall              : {recall[1]:.4f}"
    )

    print(
        f"F1                  : {f1[1]:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(
        "                    Pred Non-Off   Pred Off"
    )

    print(
        f"Actual Non-Off      "
        f"{cm[0,0]:10d}   "
        f"{cm[0,1]:10d}"
    )

    print(
        f"Actual Off          "
        f"{cm[1,0]:10d}   "
        f"{cm[1,1]:10d}"
    )

    return {
        "Dataset / Language": name,
        "Samples": len(result),
        "Accuracy": accuracy,
        "Macro-F1": macro_f1,
        "Non_Offensive_Precision": precision[0],
        "Non_Offensive_Recall": recall[0],
        "Non_Offensive_F1": f1[0],
        "Offensive_Precision": precision[1],
        "Offensive_Recall": recall[1],
        "Offensive_F1": f1[1]
    }


# ==============================================================================
# LOAD DATASET 1
# ==============================================================================

section(
    "5. PREPARING DATASET 1 - INDEPENDENT TEST SPLIT"
)

df1 = pd.read_csv(
    DATASET1
)

print(
    "Original Dataset 1 records:",
    len(df1)
)

df1["text_clean"] = (
    df1["text"]
    .apply(normalize_text)
)

df1["label_id"] = (
    df1["label"]
    .apply(normalize_dataset1_label)
)

df1 = df1[
    (df1["text_clean"].str.len() > 0)
    &
    df1["label_id"].notna()
].copy()

df1["label_id"] = (
    df1["label_id"]
    .astype(int)
)

# IMPORTANT:
# Same duplicate removal as Step 24
df1 = (
    df1
    .drop_duplicates(
        subset=["text_clean"],
        keep="first"
    )
    .reset_index(drop=True)
)

print(
    "Dataset 1 after duplicate removal:",
    len(df1)
)

# IMPORTANT:
# Reproduce EXACTLY the Step 24 Dataset-1 split:
#
# 80% train
# 10% validation
# 10% test
#
# stratified
# random_state=42

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

print()
print("Reproduced Step 24 Dataset 1 split:")
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

print()
print("Dataset 1 TEST label distribution:")
print(
    df1_test["label_id"]
    .value_counts()
    .sort_index()
)


# ==============================================================================
# DATASET 1 LEAKAGE CHECK
# ==============================================================================

section(
    "6. DATASET 1 TEST LEAKAGE CHECK"
)

train_texts = normalized_text_set(
    df1_train["text_clean"]
)

val_texts = normalized_text_set(
    df1_val["text_clean"]
)

test_texts = normalized_text_set(
    df1_test["text_clean"]
)

train_test_overlap = (
    train_texts & test_texts
)

val_test_overlap = (
    val_texts & test_texts
)

train_val_overlap = (
    train_texts & val_texts
)

print(
    "Train ↔ Validation overlap:",
    len(train_val_overlap)
)

print(
    "Train ↔ Test overlap:",
    len(train_test_overlap)
)

print(
    "Validation ↔ Test overlap:",
    len(val_test_overlap)
)

if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):

    raise RuntimeError(
        "Dataset 1 leakage detected."
    )

print(
    "DATASET 1 LEAKAGE CHECK PASSED."
)


# ==============================================================================
# LOAD DATASET 2
# ==============================================================================

section(
    "7. PREPARING DATASET 2 - INDEPENDENT TEST SPLIT"
)

df2 = pd.read_csv(
    DATASET2
)

print(
    "Original Dataset 2 records:",
    len(df2)
)

required_columns = {
    "text",
    "mapped_label",
    "language",
    "split"
}

missing_columns = (
    required_columns
    -
    set(df2.columns)
)

if missing_columns:

    raise ValueError(
        f"Dataset 2 missing columns: "
        f"{missing_columns}"
    )

df2["text_clean"] = (
    df2["text"]
    .apply(normalize_text)
)

df2["label_id"] = (
    df2["mapped_label"]
    .apply(normalize_dataset2_label)
)

df2["language_clean"] = (
    df2["language"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df2["split_clean"] = (
    df2["split"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

df2 = df2[
    (df2["text_clean"].str.len() > 0)
    &
    df2["label_id"].notna()
    &
    df2["language_clean"].isin(LANGUAGES)
].copy()

df2["label_id"] = (
    df2["label_id"]
    .astype(int)
)

# Same duplicate removal as Step 24
df2 = (
    df2
    .drop_duplicates(
        subset=["text_clean"],
        keep="first"
    )
    .reset_index(drop=True)
)

print(
    "Dataset 2 after duplicate removal:",
    len(df2)
)


# ==============================================================================
# RECREATE DATASET 2 TRAIN / VALIDATION / TEST CLEANING
# ==============================================================================

df2_train = df2[
    df2["split_clean"] == "train"
].copy()

df2_val = df2[
    df2["split_clean"] == "validation"
].copy()

df2_test = df2[
    df2["split_clean"] == "test"
].copy()


# Same cross-split leakage cleaning as Step 24

train_texts_d2 = normalized_text_set(
    df2_train["text_clean"]
)

df2_val = df2_val[
    ~df2_val["text_clean"]
    .str.lower()
    .str.strip()
    .isin(train_texts_d2)
].copy()

val_texts_d2 = normalized_text_set(
    df2_val["text_clean"]
)

df2_test = df2_test[
    ~df2_test["text_clean"]
    .str.lower()
    .str.strip()
    .isin(train_texts_d2)
    &
    ~df2_test["text_clean"]
    .str.lower()
    .str.strip()
    .isin(val_texts_d2)
].copy()


print()
print("Dataset 2 cleaned split sizes:")
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

print()
print("Dataset 2 TEST language distribution:")
print(
    df2_test["language_clean"]
    .value_counts()
)

print()
print("Dataset 2 TEST label distribution:")
print(
    df2_test["label_id"]
    .value_counts()
    .sort_index()
)


# ==============================================================================
# DATASET 2 LEAKAGE CHECK
# ==============================================================================

section(
    "8. DATASET 2 LEAKAGE CHECK"
)

train_texts_d2 = normalized_text_set(
    df2_train["text_clean"]
)

val_texts_d2 = normalized_text_set(
    df2_val["text_clean"]
)

test_texts_d2 = normalized_text_set(
    df2_test["text_clean"]
)

print(
    "Train ↔ Validation overlap:",
    len(
        train_texts_d2 & val_texts_d2
    )
)

print(
    "Train ↔ Test overlap:",
    len(
        train_texts_d2 & test_texts_d2
    )
)

print(
    "Validation ↔ Test overlap:",
    len(
        val_texts_d2 & test_texts_d2
    )
)

if (
    train_texts_d2 & val_texts_d2
    or
    train_texts_d2 & test_texts_d2
    or
    val_texts_d2 & test_texts_d2
):

    raise RuntimeError(
        "Dataset 2 leakage detected."
    )

print(
    "DATASET 2 LEAKAGE CHECK PASSED."
)


# ==============================================================================
# CROSS-DATASET TEST OVERLAP AUDIT
# ==============================================================================

section(
    "9. CROSS-DATASET TEST OVERLAP AUDIT"
)

d1_test_texts = normalized_text_set(
    df1_test["text_clean"]
)

d2_test_normalized = (
    df2_test["text_clean"]
    .str.lower()
    .str.strip()
)

cross_test_overlap = (
    d2_test_normalized
    .isin(d1_test_texts)
)

cross_test_overlap_count = int(
    cross_test_overlap.sum()
)

print(
    "Dataset 1 TEST ↔ Dataset 2 TEST overlap:",
    cross_test_overlap_count
)

if cross_test_overlap_count == 0:

    print(
        "CROSS-DATASET TEST OVERLAP CHECK PASSED."
    )

else:

    print(
        "WARNING: Cross-dataset test overlap exists."
    )

    print(
        "Overlapping records will be excluded from "
        "the combined independent-test evaluation."
    )

    df2_test = df2_test[
        ~cross_test_overlap
    ].copy()

    print(
        "Dataset 2 TEST after cross-dataset "
        "overlap removal:",
        len(df2_test)
    )


# ==============================================================================
# FINAL EVALUATION - DATASET 1 TEST
# ==============================================================================

section(
    "10. FINAL STEP 24 EVALUATION - DATASET 1 TEST"
)

print(
    "IMPORTANT: Only the independent Dataset 1 TEST "
    "split is evaluated."
)

english_result = predict_final(
    df1_test
)

english_metrics = calculate_metrics(
    "DATASET 1 - ENGLISH TEST",
    english_result
)

english_output = os.path.join(
    OUTPUT_DIR,
    "dataset1_english_test_predictions.csv"
)

english_result.to_csv(
    english_output,
    index=False
)

print(
    "\nSaved:",
    english_output
)


# ==============================================================================
# FINAL EVALUATION - DRAVIDIAN LANGUAGES
# ==============================================================================

section(
    "11. FINAL STEP 24 EVALUATION - DRAVIDIAN TEST"
)

all_metrics = [
    english_metrics
]

language_results = []

for language in LANGUAGES:

    subset = df2_test[
        df2_test["language_clean"]
        == language
    ].copy()

    if len(subset) == 0:

        print(
            f"\nSkipping {language}: no test samples."
        )

        continue

    print()
    print(
        f"Evaluating {language} TEST..."
    )

    result = predict_final(
        subset
    )

    metrics = calculate_metrics(
        f"DATASET 2 - {language} TEST",
        result
    )

    all_metrics.append(
        metrics
    )

    language_results.append(
        result
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"dataset2_{language.lower()}_test_predictions.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        "\nSaved:",
        output_file
    )


# ==============================================================================
# COMBINED DRAVIDIAN TEST
# ==============================================================================

section(
    "12. COMBINED DRAVIDIAN TEST EVALUATION"
)

if language_results:

    combined_dravidian = pd.concat(
        language_results,
        ignore_index=True
    )

    dravidian_metrics = calculate_metrics(
        "DATASET 2 - DRAVIDIAN COMBINED TEST",
        combined_dravidian
    )

    all_metrics.append(
        dravidian_metrics
    )

    combined_output = os.path.join(
        OUTPUT_DIR,
        "dataset2_dravidian_combined_test_predictions.csv"
    )

    combined_dravidian.to_csv(
        combined_output,
        index=False
    )

    print(
        "\nSaved:",
        combined_output
    )


# ==============================================================================
# OVERALL INDEPENDENT TEST EVALUATION
# ==============================================================================

section(
    "13. OVERALL INDEPENDENT TEST EVALUATION"
)

frames_for_overall = [
    english_result
]

frames_for_overall.extend(
    language_results
)

overall_result = pd.concat(
    frames_for_overall,
    ignore_index=True
)

overall_metrics = calculate_metrics(
    "OVERALL INDEPENDENT TEST",
    overall_result
)

all_metrics.append(
    overall_metrics
)

overall_output = os.path.join(
    OUTPUT_DIR,
    "overall_independent_test_predictions.csv"
)

overall_result.to_csv(
    overall_output,
    index=False
)

print(
    "\nSaved:",
    overall_output
)


# ==============================================================================
# FINAL SUMMARY TABLE
# ==============================================================================

section(
    "14. FINAL STEP 24 RESULTS SUMMARY"
)

summary_rows = []

for metrics in all_metrics:

    summary_rows.append(
        {
            "Dataset / Language":
                metrics["Dataset / Language"],

            "Samples":
                metrics["Samples"],

            "Accuracy":
                round(
                    metrics["Accuracy"],
                    4
                ),

            "Macro-F1":
                round(
                    metrics["Macro-F1"],
                    4
                ),

            "Offensive Precision":
                round(
                    metrics["Offensive_Precision"],
                    4
                ),

            "Offensive Recall":
                round(
                    metrics["Offensive_Recall"],
                    4
                ),

            "Offensive F1":
                round(
                    metrics["Offensive_F1"],
                    4
                )
        }
    )

summary_df = pd.DataFrame(
    summary_rows
)

print(
    summary_df.to_string(
        index=False
    )
)

summary_output = os.path.join(
    OUTPUT_DIR,
    "final_independent_test_summary.csv"
)

summary_df.to_csv(
    summary_output,
    index=False
)

print(
    "\nSummary saved:",
    summary_output
)


# ==============================================================================
# FINAL CONFIGURATION
# ==============================================================================

section(
    "15. FINAL SYSTEM CONFIGURATION"
)

print(
    "Frozen Step 20 XLM-R"
)

print(
    "Mean + Max contextual representation"
)

print(
    "Current Step 24 semantic classifier"
)

print(
    "Validation-calibrated hybrid decision"
)

print(
    f"\nXLM-R weight    : {XLMR_WEIGHT:.4f}"
)

print(
    f"Semantic weight : {SEMANTIC_WEIGHT:.4f}"
)

print(
    f"Threshold       : {THRESHOLD:.4f}"
)

print(
    "\nNo offensive vocabulary is hardcoded."
)

print(
    "No keyword rules are used."
)

print(
    "No manual sentence rules are used."
)

print(
    "Step 20 was NOT retrained."
)

print(
    "Step 24 semantic classifier was NOT retrained."
)

print(
    "Dataset 1 TEST is an independent held-out split."
)

print(
    "Dataset 2 TEST is the independent official test split."
)

print(
    "\nFINAL EVALUATION COMPLETE."
)

print(
    "\nResults directory:"
)

print(
    OUTPUT_DIR
)