"""
==============================================================================
STEP 31 — EMOJI-AWARE EXPERIMENT
==============================================================================

Purpose:
    Separate experimental branch to test whether simple emoji features
    provide useful signal for offensive-language classification.

Important:
    - Step 20 is NOT retrained.
    - Step 24 is NOT retrained.
    - Step 26 is NOT retrained.
    - Existing Step 26 independent-test predictions are used as the
      frozen baseline.
    - XLM-R representations are NOT recomputed.
    - This experiment is separate from the final multilingual pipeline.

Baseline:
    Existing Step 26 independent-test predictions.

Experimental branch:
    Emoji-only Logistic Regression using six emoji features.

Output:
    results\emoji_analysis\step31_emoji_experiment
==============================================================================
"""

import os
import re
import json
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

ROOT = r"E:\Offensive Language Detection in Social Media_NLP"

DATA_FILE = os.path.join(
    ROOT,
    "data",
    "dravidian_clean",
    "dravidian_multilingual_clean.csv",
)

STEP24_ROOT = os.path.join(
    ROOT,
    "results",
    "multilingual_cross_dataset_final",
    "step24_semantic_final",
)

CALIBRATION_FILE = os.path.join(
    STEP24_ROOT,
    "hybrid_calibration.json",
)

STEP26_FINAL_DIR = os.path.join(
    STEP24_ROOT,
    "final_evaluation",
)

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "emoji_analysis",
    "step31_emoji_experiment",
)

os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42

EXPECTED_INITIAL_TRAIN = 48743
EXPECTED_INITIAL_VALIDATION = 5651
EXPECTED_INITIAL_TEST = 5580

EXPECTED_FINAL_TRAIN = 48743
EXPECTED_FINAL_VALIDATION = 5643
EXPECTED_FINAL_TEST = 5571


# ==============================================================================
# 2. HELPERS
# ==============================================================================

def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def leakage_key(text):
    """
    Same case-insensitive/strip logic used for split-overlap checking.
    This is ONLY for leakage checking.

    It is deliberately NOT used to align Step 26 prediction rows because
    distinct social-media texts can collapse under case-insensitive keys.
    """
    if pd.isna(text):
        return ""

    return str(text).lower().strip()


def normalize_label(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value in {
        "0",
        "non_offensive",
        "non-offensive",
        "non offensive",
        "not offensive",
        "nonoffensive",
    }:
        return 0

    if value in {
        "1",
        "offensive",
    }:
        return 1

    return np.nan


def normalize_language(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()

    mapping = {
        "tamil": "Tamil",
        "malayalam": "Malayalam",
        "kannada": "Kannada",
    }

    return mapping.get(value, str(value).strip())


def normalize_split(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    aliases = {
        "train": "train",
        "training": "train",

        "val": "validation",
        "valid": "validation",
        "dev": "validation",
        "validation": "validation",

        "test": "test",
        "testing": "test",
    }

    return aliases.get(value, value)


def normalized_text_set(series):
    return set(
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )


# ==============================================================================
# 3. EMOJI FEATURES
# ==============================================================================

EMOJI_RANGES = [
    (0x1F000, 0x1FAFF),
    (0x1FC00, 0x1FFFD),
    (0x2600, 0x27BF),
    (0x2300, 0x23FF),
    (0x2B00, 0x2BFF),
    (0x2934, 0x2935),
    (0x3030, 0x303D),
    (0x3297, 0x3299),
]

EMOJI_FEATURE_NAMES = [
    "emoji_count",
    "unique_emoji_count",
    "emoji_density",
    "repeated_emoji_count",
    "has_repeated_emoji",
    "max_emoji_repetition",
]


def is_emoji(character):
    codepoint = ord(character)

    for start, end in EMOJI_RANGES:
        if start <= codepoint <= end:
            return True

    if 0x1F1E6 <= codepoint <= 0x1F1FF:
        return True

    return False


def extract_emojis(text):
    if not isinstance(text, str):
        return []

    return [
        character
        for character in text
        if is_emoji(character)
    ]


def extract_emoji_features(text):
    emojis = extract_emojis(text)

    counts = Counter(emojis)

    emoji_count = len(emojis)

    unique_emoji_count = len(set(emojis))

    text_length = len(text)

    emoji_density = (
        emoji_count / text_length
        if text_length > 0
        else 0.0
    )

    repeated_emoji_count = sum(
        max(0, count - 1)
        for count in counts.values()
    )

    has_repeated_emoji = int(
        repeated_emoji_count > 0
    )

    max_emoji_repetition = (
        max(counts.values())
        if counts
        else 0
    )

    return [
        emoji_count,
        unique_emoji_count,
        emoji_density,
        repeated_emoji_count,
        has_repeated_emoji,
        max_emoji_repetition,
    ]


def build_emoji_features(frame):
    return np.asarray(
        [
            extract_emoji_features(text)
            for text in frame["text"]
        ],
        dtype=np.float64,
    )


# ==============================================================================
# 4. METRICS
# ==============================================================================

def calculate_metrics(y_true, predictions):
    return {
        "samples": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "offensive_precision": float(
            precision_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "offensive_recall": float(
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "offensive_f1": float(
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
    }


# ==============================================================================
# 5. START
# ==============================================================================

section("STEP 31 — EMOJI-AWARE EXPERIMENT")

print("Dataset:")
print(DATA_FILE)

print()
print("Output:")
print(OUT_DIR)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )


# ==============================================================================
# 6. LOAD DATASET
# ==============================================================================

df = pd.read_csv(DATA_FILE)

print()
print("Raw shape:", df.shape)
print("Columns:", list(df.columns))

required_dataset_columns = {
    "text",
    "mapped_label",
    "language",
    "split",
}

missing = required_dataset_columns - set(df.columns)

if missing:
    raise RuntimeError(
        f"Dataset is missing required columns: {sorted(missing)}"
    )


# ==============================================================================
# 7. PREPARE DATASET
# ==============================================================================

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .apply(normalize_text)
)

df["label_id"] = (
    df["mapped_label"]
    .apply(normalize_label)
)

df["language_clean"] = (
    df["language"]
    .apply(normalize_language)
)

# IMPORTANT:
# Normalize aliases BEFORE checking split sizes.
df["split_clean"] = (
    df["split"]
    .apply(normalize_split)
)

df = df[
    (df["text"].str.len() > 0)
    & df["label_id"].notna()
    & df["language_clean"].isin(
        ["Tamil", "Malayalam", "Kannada"]
    )
    & df["split_clean"].isin(
        ["train", "validation", "test"]
    )
].copy()

df["label_id"] = df["label_id"].astype(int)


# ==============================================================================
# 8. EXACT GLOBAL DEDUPLICATION
# ==============================================================================

records_before_dedup = len(df)

df = (
    df
    .drop_duplicates(
        subset=["text"],
        keep="first",
    )
    .reset_index(drop=True)
)

records_after_dedup = len(df)

print()
print("Records before deduplication:", records_before_dedup)
print("Records after deduplication :", records_after_dedup)
print(
    "Duplicates removed          :",
    records_before_dedup - records_after_dedup,
)

initial_counts = (
    df["split_clean"]
    .value_counts()
    .reindex(
        ["train", "validation", "test"],
        fill_value=0,
    )
)

print()
print("Initial split sizes:")
print("Train      :", int(initial_counts["train"]))
print("Validation :", int(initial_counts["validation"]))
print("Test       :", int(initial_counts["test"]))

if int(initial_counts["train"]) != EXPECTED_INITIAL_TRAIN:
    raise RuntimeError(
        f"Unexpected initial train size: {int(initial_counts['train'])}"
    )

if int(initial_counts["validation"]) != EXPECTED_INITIAL_VALIDATION:
    raise RuntimeError(
        f"Unexpected initial validation size: "
        f"{int(initial_counts['validation'])}"
    )

if int(initial_counts["test"]) != EXPECTED_INITIAL_TEST:
    raise RuntimeError(
        f"Unexpected initial test size: {int(initial_counts['test'])}"
    )


# ==============================================================================
# 9. REPRODUCE EXACT STEP 26 LEAKAGE CLEANING
# ==============================================================================

section("REPRODUCING EXACT STEP 26 TEST CLEANING")

train = (
    df[df["split_clean"] == "train"]
    .copy()
    .reset_index(drop=True)
)

validation = (
    df[df["split_clean"] == "validation"]
    .copy()
    .reset_index(drop=True)
)

test = (
    df[df["split_clean"] == "test"]
    .copy()
    .reset_index(drop=True)
)

train_keys = normalized_text_set(train["text"])
validation_keys = normalized_text_set(validation["text"])
test_keys = normalized_text_set(test["text"])

# Train ↔ validation
train_validation_overlap = train_keys & validation_keys

if train_validation_overlap:
    validation = validation[
        ~validation["text"].map(leakage_key).isin(
            {leakage_key(x) for x in train_validation_overlap}
        )
    ].copy().reset_index(drop=True)

# Recompute because validation changed.
train_keys = normalized_text_set(train["text"])
validation_keys = normalized_text_set(validation["text"])
test_keys = normalized_text_set(test["text"])

# Train ↔ test
train_test_overlap = train_keys & test_keys

if train_test_overlap:
    test = test[
        ~test["text"].map(leakage_key).isin(
            {leakage_key(x) for x in train_test_overlap}
        )
    ].copy().reset_index(drop=True)

# Recompute after test cleaning.
train_keys = normalized_text_set(train["text"])
validation_keys = normalized_text_set(validation["text"])
test_keys = normalized_text_set(test["text"])

# Validation ↔ test
validation_test_overlap = validation_keys & test_keys

if validation_test_overlap:
    test = test[
        ~test["text"].map(leakage_key).isin(
            {leakage_key(x) for x in validation_test_overlap}
        )
    ].copy().reset_index(drop=True)

removed_train_validation = (
    EXPECTED_INITIAL_VALIDATION - len(validation)
)

removed_train_test = (
    EXPECTED_INITIAL_TEST - len(test)
)

removed_validation_test = 0

# Recalculate final overlap sets.
train_keys = normalized_text_set(train["text"])
validation_keys = normalized_text_set(validation["text"])
test_keys = normalized_text_set(test["text"])

train_validation_overlap = train_keys & validation_keys
train_test_overlap = train_keys & test_keys
validation_test_overlap = validation_keys & test_keys

print()
print(
    "Removed train ↔ validation duplicates:",
    removed_train_validation,
)

print(
    "Removed train ↔ test duplicates:",
    removed_train_test,
)

print(
    "Removed validation ↔ test duplicates:",
    removed_validation_test,
)

print()
print("After exact Step 26 cleaning:")
print("Train      :", len(train))
print("Validation :", len(validation))
print("Test       :", len(test))

print()
print("Final leakage verification:")
print(
    "Train ↔ Validation:",
    len(train_validation_overlap),
)
print(
    "Train ↔ Test      :",
    len(train_test_overlap),
)
print(
    "Validation ↔ Test :",
    len(validation_test_overlap),
)

if (
    len(train_validation_overlap) != 0
    or len(train_test_overlap) != 0
    or len(validation_test_overlap) != 0
):
    raise RuntimeError(
        "DATA LEAKAGE CHECK FAILED"
    )

print("DATA LEAKAGE CHECK PASSED")

if len(train) != EXPECTED_FINAL_TRAIN:
    raise RuntimeError(
        f"Unexpected final train size: {len(train)}"
    )

if len(validation) != EXPECTED_FINAL_VALIDATION:
    raise RuntimeError(
        f"Unexpected final validation size: {len(validation)}"
    )

if len(test) != EXPECTED_FINAL_TEST:
    raise RuntimeError(
        f"Unexpected final test size: {len(test)}"
    )

print()
print(
    "EXACT STEP 26 TEST SIZE CONFIRMED:",
    len(test),
)

print()
print("Step 26-compatible test language distribution:")
print(test["language_clean"].value_counts())

print()
print("Step 26-compatible test label distribution:")
print(test["label_id"].value_counts().sort_index())


# ==============================================================================
# 10. LOAD FROZEN STEP 24 CALIBRATION
# ==============================================================================

section("LOADING FROZEN STEP 24 CALIBRATION")

if not os.path.exists(CALIBRATION_FILE):
    raise FileNotFoundError(
        f"Calibration file not found:\n{CALIBRATION_FILE}"
    )

with open(
    CALIBRATION_FILE,
    "r",
    encoding="utf-8",
) as file:
    calibration = json.load(file)

step24_threshold = float(
    calibration["threshold"]
)

step24_xlmr_weight = float(
    calibration["xlmr_weight"]
)

step24_semantic_weight = float(
    calibration["semantic_weight"]
)

print(
    "Step 24 threshold:",
    step24_threshold,
)

print(
    "Step 24 XLM-R weight:",
    step24_xlmr_weight,
)

print(
    "Step 24 semantic weight:",
    step24_semantic_weight,
)

if not np.isclose(
    step24_xlmr_weight + step24_semantic_weight,
    1.0,
):
    raise RuntimeError(
        "Step 24 calibration weights do not sum to 1."
    )


# ==============================================================================
# 11. LOAD EXISTING STEP 26 PREDICTIONS
# ==============================================================================

section("LOADING EXISTING STEP 26 PREDICTIONS")

prediction_files = {
    "Tamil": os.path.join(
        STEP26_FINAL_DIR,
        "dataset2_tamil_test_predictions.csv",
    ),
    "Malayalam": os.path.join(
        STEP26_FINAL_DIR,
        "dataset2_malayalam_test_predictions.csv",
    ),
    "Kannada": os.path.join(
        STEP26_FINAL_DIR,
        "dataset2_kannada_test_predictions.csv",
    ),
}

prediction_frames = []

for language, path in prediction_files.items():

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing Step 26 prediction file for "
            f"{language}:\n{path}"
        )

    print(language, "->", path)

    part = pd.read_csv(path)

    print(
        language,
        "prediction rows:",
        len(part),
    )

    required_prediction_columns = {
        "text",
        "label_id",
        "language",
        "hybrid_probability",
    }

    missing_columns = (
        required_prediction_columns
        - set(part.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"{language} prediction file is missing: "
            f"{sorted(missing_columns)}"
        )

    part = part.copy()

    part["expected_language"] = language

    part["prediction_label"] = (
        part["label_id"]
        .apply(normalize_label)
    )

    part["baseline_probability"] = pd.to_numeric(
        part["hybrid_probability"],
        errors="coerce",
    )

    if part["baseline_probability"].isna().any():
        raise RuntimeError(
            f"Invalid Step 26 hybrid probabilities "
            f"detected in {language} file."
        )

    if len(part) != int(
        (test["language_clean"] == language).sum()
    ):
        raise RuntimeError(
            f"Step 26 {language} prediction row count "
            f"does not match reconstructed test count."
        )

    prediction_frames.append(part)

predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

print()
print(
    "Total Step 26 prediction rows:",
    len(predictions),
)

predictions["baseline_probability"] = pd.to_numeric(
    predictions["hybrid_probability"],
    errors="coerce",
)

if predictions["baseline_probability"].isna().any():
    raise RuntimeError(
        "Invalid Step 26 hybrid probabilities detected."
    )

if len(predictions) != EXPECTED_FINAL_TEST:
    raise RuntimeError(
        "Step 26 prediction files do not contain "
        f"the expected {EXPECTED_FINAL_TEST} rows. "
        f"Found: {len(predictions)}"
    )


# ==============================================================================
# 12. CORRECT STEP 26 PREDICTION ALIGNMENT
# ==============================================================================

section("ALIGNING FROZEN STEP 26 PREDICTIONS")

"""
IMPORTANT:

Step 26 prediction CSVs contain:
    label_id          = ACTUAL ground-truth label
    predicted_label   = MODEL prediction
    hybrid_probability = calibrated model probability

Therefore label_id MUST NOT be used as the model prediction.

This version uses the exact `text_clean` field saved by Step 26 as the
primary alignment key, together with language. It then reconstructs the
actual Step 26 prediction from:

    hybrid_probability >= Step 24 threshold

This is necessary because the earlier Step 31 version accidentally treated
label_id as the prediction, which produced a false 1.0000 baseline.
"""

test = test.copy()

# --------------------------------------------------------------------------
# Prepare prediction files
# --------------------------------------------------------------------------

prepared_prediction_parts = []

for language in ["Tamil", "Malayalam", "Kannada"]:

    part = predictions[
        predictions["expected_language"] == language
    ].copy().reset_index(drop=True)

    if "text_clean" not in part.columns:
        raise RuntimeError(
            f"{language} Step 26 prediction file does not contain "
            f"`text_clean`.\nAvailable columns: {list(part.columns)}"
        )

    part["step26_text_clean"] = (
        part["text_clean"]
        .fillna("")
        .astype(str)
        .apply(normalize_text)
    )

    part["step26_language"] = language

    # label_id is ACTUAL ground truth.
    part["step26_actual_label"] = (
        part["label_id"]
        .apply(normalize_label)
    )

    # Reconstruct the TRUE Step 26 MODEL prediction.
    part["step26_predicted_label"] = (
        part["baseline_probability"] >= step24_threshold
    ).astype(int)

    if part["step26_actual_label"].isna().any():
        raise RuntimeError(
            f"Invalid actual labels in Step 26 {language} predictions."
        )

    prepared_prediction_parts.append(part)

predictions = pd.concat(
    prepared_prediction_parts,
    ignore_index=True,
)

# --------------------------------------------------------------------------
# Exact Step 26 cleaned test key
# --------------------------------------------------------------------------

test["step26_text_clean"] = (
    test["text"]
    .fillna("")
    .astype(str)
    .apply(normalize_text)
)

test["step26_language"] = test["language_clean"]

test["match_key"] = (
    test["step26_language"].astype(str)
    + "\t"
    + test["step26_text_clean"].astype(str)
)

predictions["match_key"] = (
    predictions["step26_language"].astype(str)
    + "\t"
    + predictions["step26_text_clean"].astype(str)
)

test_duplicate_keys = int(
    test["match_key"].duplicated(keep=False).sum()
)

prediction_duplicate_keys = int(
    predictions["match_key"].duplicated(keep=False).sum()
)

print()
print(
    "Duplicate exact Step 26 test keys:",
    test_duplicate_keys,
)
print(
    "Duplicate exact Step 26 prediction keys:",
    prediction_duplicate_keys,
)

if test_duplicate_keys != 0:
    raise RuntimeError(
        "Duplicate exact cleaned-text keys exist in reconstructed "
        "Step 26 test set."
    )

if prediction_duplicate_keys != 0:
    raise RuntimeError(
        "Duplicate exact cleaned-text keys exist in Step 26 "
        "prediction files."
    )

# --------------------------------------------------------------------------
# Exact one-to-one alignment
# --------------------------------------------------------------------------

merged = test.merge(
    predictions[
        [
            "match_key",
            "step26_actual_label",
            "step26_predicted_label",
            "baseline_probability",
        ]
    ],
    on="match_key",
    how="left",
    validate="one_to_one",
)

matched_rows = int(
    merged["step26_predicted_label"].notna().sum()
)

print()
print("Expected test rows:", len(test))
print("Matched prediction rows:", matched_rows)

if matched_rows != EXPECTED_FINAL_TEST:
    print()
    print("First unmatched rows:")
    print(
        merged[
            merged["step26_predicted_label"].isna()
        ][
            [
                "language_clean",
                "text",
                "step26_text_clean",
            ]
        ].head(20).to_string(index=False)
    )

    raise RuntimeError(
        "\nExact Step 26 prediction matching failed.\n"
        f"Expected: {EXPECTED_FINAL_TEST}\n"
        f"Matched : {matched_rows}\n"
        "Step 26 text_clean does not match the reconstructed test set."
    )

# --------------------------------------------------------------------------
# Verify actual labels
# --------------------------------------------------------------------------

label_mismatch = int(
    (
        merged["label_id"].astype(int).to_numpy()
        !=
        merged["step26_actual_label"].astype(int).to_numpy()
    ).sum()
)

print(
    "Actual-label mismatches:",
    label_mismatch,
)

if label_mismatch != 0:
    raise RuntimeError(
        "Step 26 actual labels do not match reconstructed test labels."
    )

# --------------------------------------------------------------------------
# Store TRUE Step 26 predictions
# --------------------------------------------------------------------------

test["baseline_prediction"] = (
    merged["step26_predicted_label"]
    .astype(int)
    .to_numpy()
)

test["baseline_probability"] = (
    merged["baseline_probability"]
    .astype(float)
    .to_numpy()
)

if (
    (test["baseline_probability"] < 0).any()
    or
    (test["baseline_probability"] > 1).any()
):
    raise RuntimeError(
        "Step 26 hybrid probabilities outside [0,1]."
    )

print()
print(
    "STEP 26 PREDICTIONS ALIGNED:",
    len(test),
)

alignment_report = []

for language in ["Tamil", "Malayalam", "Kannada"]:

    alignment_report.append(
        {
            "language": language,
            "test_rows": int(
                (test["language_clean"] == language).sum()
            ),
            "prediction_rows": int(
                (predictions["step26_language"] == language).sum()
            ),
            "actual_label_mismatches": 0,
            "alignment_method":
                "exact Step 26 text_clean + language",
        }
    )

alignment_report_df = pd.DataFrame(
    alignment_report
)

alignment_report_df.to_csv(
    os.path.join(
        OUT_DIR,
        "step26_alignment_report.csv",
    ),
    index=False,
    encoding="utf-8-sig",
)

print()
print(
    alignment_report_df.to_string(index=False)
)


# ==============================================================================
# 13. BASELINE METRICS
# ==============================================================================
# ==============================================================================
# 13. BASELINE METRICS
# ==============================================================================

section("FROZEN STEP 26 BASELINE METRICS")

y_test = test["label_id"].astype(int).to_numpy()

baseline_predictions = (
    test["baseline_prediction"]
    .astype(int)
    .to_numpy()
)

baseline_probabilities = (
    test["baseline_probability"]
    .astype(float)
    .to_numpy()
)

baseline_metrics = calculate_metrics(
    y_test,
    baseline_predictions,
)

print(
    json.dumps(
        baseline_metrics,
        indent=2,
    )
)

baseline_cm = confusion_matrix(
    y_test,
    baseline_predictions,
    labels=[0, 1],
)

print()
print("Frozen baseline confusion matrix:")
print(baseline_cm)

# Step 26 already produced these independent-test metrics.
# This sanity check prevents an accidental column/alignment error from
# generating fake baseline metrics.
EXPECTED_STEP26_ACCURACY = 0.7859
EXPECTED_STEP26_MACRO_F1 = 0.6889

if not np.isclose(
    baseline_metrics["accuracy"],
    EXPECTED_STEP26_ACCURACY,
    atol=0.002,
):
    raise RuntimeError(
        "Step 26 baseline sanity check FAILED. "
        f"Expected approximately {EXPECTED_STEP26_ACCURACY:.4f} accuracy, "
        f"got {baseline_metrics['accuracy']:.4f}."
    )

if not np.isclose(
    baseline_metrics["macro_f1"],
    EXPECTED_STEP26_MACRO_F1,
    atol=0.002,
):
    raise RuntimeError(
        "Step 26 baseline sanity check FAILED. "
        f"Expected approximately {EXPECTED_STEP26_MACRO_F1:.4f} macro-F1, "
        f"got {baseline_metrics['macro_f1']:.4f}."
    )

print()
print("STEP 26 BASELINE SANITY CHECK PASSED")
print(
    f"Accuracy : {baseline_metrics['accuracy']:.4f}"
)
print(
    f"Macro-F1 : {baseline_metrics['macro_f1']:.4f}"
)


# ==============================================================================
# 14. BUILD EMOJI FEATURES
# ==============================================================================

section("BUILDING EMOJI FEATURES")

print("Features:")
for feature in EMOJI_FEATURE_NAMES:
    print("  -", feature)

X_train = build_emoji_features(train)
X_validation = build_emoji_features(validation)
X_test = build_emoji_features(test)

y_train = (
    train["label_id"]
    .astype(int)
    .to_numpy()
)

y_validation = (
    validation["label_id"]
    .astype(int)
    .to_numpy()
)

print()
print("Train feature shape      :", X_train.shape)
print("Validation feature shape :", X_validation.shape)
print("Test feature shape       :", X_test.shape)


# ==============================================================================
# 15. TRAIN EMOJI-ONLY CLASSIFIER
# ==============================================================================

section("TRAINING EMOJI-ONLY EXPERIMENTAL BRANCH")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_validation_scaled = scaler.transform(
    X_validation
)

X_test_scaled = scaler.transform(
    X_test
)

emoji_classifier = LogisticRegression(
    C=1.0,
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

emoji_classifier.fit(
    X_train_scaled,
    y_train,
)

print()
print("Emoji-only classifier trained.")
print("No XLM-R retraining performed.")
print("No Step 20/24/26 modification performed.")


# ==============================================================================
# 16. VALIDATION THRESHOLD SEARCH
# ==============================================================================

section("VALIDATION THRESHOLD SEARCH")

validation_probabilities = (
    emoji_classifier
    .predict_proba(X_validation_scaled)[:, 1]
)

threshold_rows = []

for threshold in np.arange(
    0.30,
    0.701,
    0.01,
):

    threshold = float(
        round(threshold, 2)
    )

    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    threshold_rows.append(
        {
            "threshold": threshold,
            "accuracy": accuracy_score(
                y_validation,
                validation_predictions,
            ),
            "macro_f1": f1_score(
                y_validation,
                validation_predictions,
                average="macro",
                zero_division=0,
            ),
            "offensive_precision": precision_score(
                y_validation,
                validation_predictions,
                pos_label=1,
                zero_division=0,
            ),
            "offensive_recall": recall_score(
                y_validation,
                validation_predictions,
                pos_label=1,
                zero_division=0,
            ),
            "offensive_f1": f1_score(
                y_validation,
                validation_predictions,
                pos_label=1,
                zero_division=0,
            ),
        }
    )

threshold_df = pd.DataFrame(
    threshold_rows
)

best_row = (
    threshold_df
    .sort_values(
        [
            "macro_f1",
            "offensive_f1",
            "accuracy",
        ],
        ascending=False,
    )
    .iloc[0]
)

best_emoji_threshold = float(
    best_row["threshold"]
)

print()
print(
    "Best emoji threshold:",
    best_emoji_threshold,
)

print(
    "Validation macro-F1:",
    float(best_row["macro_f1"]),
)


# ==============================================================================
# 17. INDEPENDENT TEST EVALUATION
# ==============================================================================

section("INDEPENDENT TEST EVALUATION")

emoji_test_probabilities = (
    emoji_classifier
    .predict_proba(X_test_scaled)[:, 1]
)

emoji_test_predictions = (
    emoji_test_probabilities >= best_emoji_threshold
).astype(int)

emoji_metrics = calculate_metrics(
    y_test,
    emoji_test_predictions,
)

emoji_cm = confusion_matrix(
    y_test,
    emoji_test_predictions,
    labels=[0, 1],
)

print()
print("Emoji experimental branch metrics:")
print(
    json.dumps(
        emoji_metrics,
        indent=2,
    )
)

print()
print("Emoji confusion matrix:")
print(emoji_cm)


# ==============================================================================
# 18. LANGUAGE-WISE COMPARISON
# ==============================================================================

section("LANGUAGE-WISE COMPARISON")

language_rows = []

for language in ["Tamil", "Malayalam", "Kannada"]:

    mask = (
        test["language_clean"]
        .to_numpy()
        == language
    )

    language_y = y_test[mask]

    language_baseline = (
        baseline_predictions[mask]
    )

    language_emoji = (
        emoji_test_predictions[mask]
    )

    baseline_language_metrics = calculate_metrics(
        language_y,
        language_baseline,
    )

    emoji_language_metrics = calculate_metrics(
        language_y,
        language_emoji,
    )

    language_rows.append(
        {
            "language": language,
            "samples": int(mask.sum()),

            "baseline_accuracy":
                baseline_language_metrics["accuracy"],
            "baseline_macro_f1":
                baseline_language_metrics["macro_f1"],
            "baseline_offensive_precision":
                baseline_language_metrics[
                    "offensive_precision"
                ],
            "baseline_offensive_recall":
                baseline_language_metrics[
                    "offensive_recall"
                ],
            "baseline_offensive_f1":
                baseline_language_metrics[
                    "offensive_f1"
                ],

            "emoji_accuracy":
                emoji_language_metrics["accuracy"],
            "emoji_macro_f1":
                emoji_language_metrics["macro_f1"],
            "emoji_offensive_precision":
                emoji_language_metrics[
                    "offensive_precision"
                ],
            "emoji_offensive_recall":
                emoji_language_metrics[
                    "offensive_recall"
                ],
            "emoji_offensive_f1":
                emoji_language_metrics[
                    "offensive_f1"
                ],
        }
    )

language_df = pd.DataFrame(
    language_rows
)

print()
print(
    language_df.to_string(index=False)
)


# ==============================================================================
# 19. EMOJI VS NON-EMOJI SUBSET ANALYSIS
# ==============================================================================

section("EMOJI VS NON-EMOJI SUBSET ANALYSIS")

has_emoji_mask = (
    X_test[:, 0] > 0
)

subset_rows = []

for subset_name, mask in [
    ("Emoji-containing", has_emoji_mask),
    ("No-emoji", ~has_emoji_mask),
]:

    subset_y = y_test[mask]

    subset_baseline = (
        baseline_predictions[mask]
    )

    subset_emoji = (
        emoji_test_predictions[mask]
    )

    baseline_subset_metrics = calculate_metrics(
        subset_y,
        subset_baseline,
    )

    emoji_subset_metrics = calculate_metrics(
        subset_y,
        subset_emoji,
    )

    subset_rows.append(
        {
            "subset": subset_name,
            "samples": int(mask.sum()),

            "baseline_accuracy":
                baseline_subset_metrics["accuracy"],
            "baseline_macro_f1":
                baseline_subset_metrics["macro_f1"],
            "baseline_offensive_f1":
                baseline_subset_metrics["offensive_f1"],

            "emoji_accuracy":
                emoji_subset_metrics["accuracy"],
            "emoji_macro_f1":
                emoji_subset_metrics["macro_f1"],
            "emoji_offensive_f1":
                emoji_subset_metrics["offensive_f1"],
        }
    )

subset_df = pd.DataFrame(
    subset_rows
)

print()
print(
    subset_df.to_string(index=False)
)


# ==============================================================================
# 20. OVERALL MODEL COMPARISON
# ==============================================================================

section("MODEL COMPARISON")

comparison_df = pd.DataFrame(
    [
        {
            "model": "Frozen Step 26 Baseline",
            **baseline_metrics,
        },
        {
            "model": "Emoji Experimental Branch",
            **emoji_metrics,
        },
    ]
)

comparison_df["delta_accuracy_vs_baseline"] = (
    comparison_df["accuracy"]
    - baseline_metrics["accuracy"]
)

comparison_df["delta_macro_f1_vs_baseline"] = (
    comparison_df["macro_f1"]
    - baseline_metrics["macro_f1"]
)

comparison_df["delta_offensive_precision_vs_baseline"] = (
    comparison_df["offensive_precision"]
    - baseline_metrics["offensive_precision"]
)

comparison_df["delta_offensive_recall_vs_baseline"] = (
    comparison_df["offensive_recall"]
    - baseline_metrics["offensive_recall"]
)

comparison_df["delta_offensive_f1_vs_baseline"] = (
    comparison_df["offensive_f1"]
    - baseline_metrics["offensive_f1"]
)

print()
print(
    comparison_df.to_string(index=False)
)


# ==============================================================================
# 21. TEST PREDICTIONS
# ==============================================================================

prediction_output = test[
    [
        "text",
        "language_clean",
        "label_id",
        "baseline_prediction",
        "baseline_probability",
    ]
].copy()

prediction_output["emoji_probability"] = (
    emoji_test_probabilities
)

prediction_output["emoji_prediction"] = (
    emoji_test_predictions
)

prediction_output["has_emoji"] = (
    X_test[:, 0] > 0
).astype(int)

prediction_output["emoji_count"] = (
    X_test[:, 0]
)

prediction_output["unique_emoji_count"] = (
    X_test[:, 1]
)

prediction_output["emoji_density"] = (
    X_test[:, 2]
)

prediction_output["repeated_emoji_count"] = (
    X_test[:, 3]
)

prediction_output["has_repeated_emoji"] = (
    X_test[:, 4]
)

prediction_output["max_emoji_repetition"] = (
    X_test[:, 5]
)

prediction_output["baseline_correct"] = (
    prediction_output["baseline_prediction"]
    == prediction_output["label_id"]
)

prediction_output["emoji_correct"] = (
    prediction_output["emoji_prediction"]
    == prediction_output["label_id"]
)

prediction_output["emoji_improved_error"] = (
    (~prediction_output["baseline_correct"])
    & prediction_output["emoji_correct"]
)

prediction_output["emoji_introduced_error"] = (
    prediction_output["baseline_correct"]
    & (~prediction_output["emoji_correct"])
)

prediction_output.to_csv(
    os.path.join(
        OUT_DIR,
        "emoji_test_predictions.csv",
    ),
    index=False,
    encoding="utf-8-sig",
)


# ==============================================================================
# 22. FIXED / INTRODUCED ERRORS
# ==============================================================================

fixed_errors = prediction_output[
    prediction_output["emoji_improved_error"]
].copy()

introduced_errors = prediction_output[
    prediction_output["emoji_introduced_error"]
].copy()

fixed_errors.to_csv(
    os.path.join(
        OUT_DIR,
        "emoji_fixed_errors.csv",
    ),
    index=False,
    encoding="utf-8-sig",
)

introduced_errors.to_csv(
    os.path.join(
        OUT_DIR,
        "emoji_introduced_errors.csv",
    ),
    index=False,
    encoding="utf-8-sig",
)

print()
print(
    "Errors corrected by emoji branch:",
    len(fixed_errors),
)

print(
    "Errors introduced by emoji branch:",
    len(introduced_errors),
)


# ==============================================================================
# 23. SAVE MODEL
# ==============================================================================

joblib.dump(
    {
        "scaler": scaler,
        "classifier": emoji_classifier,
        "feature_names": EMOJI_FEATURE_NAMES,
        "threshold": best_emoji_threshold,
    },
    os.path.join(
        OUT_DIR,
        "emoji_only_classifier.joblib",
    ),
)


# ==============================================================================
# 24. SAVE RESULTS
# ==============================================================================

comparison_df.to_csv(
    os.path.join(
        OUT_DIR,
        "emoji_model_comparison.csv",
    ),
    index=False,
    encoding="utf-8-sig",
)

language_df.to_csv(
    os.path.join(
        OUT_DIR,
        "emoji_language_comparison.csv",
    ),
    index=False,
    encoding="utf-8-sig",
)

subset_df.to_csv(
    os.path.join(
        OUT_DIR,
        "emoji_vs_nonemoji_comparison.csv",
    ),
    index=False,
    encoding="utf-8-sig",
)

threshold_df.to_csv(
    os.path.join(
        OUT_DIR,
        "emoji_validation_threshold_search.csv",
    ),
    index=False,
    encoding="utf-8-sig",
)

baseline_cm_df = pd.DataFrame(
    baseline_cm,
    index=[
        "Actual Non-Offensive",
        "Actual Offensive",
    ],
    columns=[
        "Predicted Non-Offensive",
        "Predicted Offensive",
    ],
)

emoji_cm_df = pd.DataFrame(
    emoji_cm,
    index=[
        "Actual Non-Offensive",
        "Actual Offensive",
    ],
    columns=[
        "Predicted Non-Offensive",
        "Predicted Offensive",
    ],
)

baseline_cm_df.to_csv(
    os.path.join(
        OUT_DIR,
        "baseline_confusion_matrix.csv",
    ),
    encoding="utf-8-sig",
)

emoji_cm_df.to_csv(
    os.path.join(
        OUT_DIR,
        "emoji_confusion_matrix.csv",
    ),
    encoding="utf-8-sig",
)


# ==============================================================================
# 25. SAVE CONFIGURATION
# ==============================================================================

config = {
    "experiment":
        "Step 31 Emoji-Aware Experimental Branch",

    "dataset":
        DATA_FILE,

    "output_directory":
        OUT_DIR,

    "step20_retrained":
        False,

    "step24_retrained":
        False,

    "step26_retrained":
        False,

    "xlmr_recomputed":
        False,

    "baseline_source":
        "Existing Step 26 independent-test prediction files",

    "baseline_alignment":
        "Language-wise Step 26 test-row order; labels verified position-by-position",

    "emoji_features":
        EMOJI_FEATURE_NAMES,

    "step24_threshold":
        step24_threshold,

    "step24_xlmr_weight":
        step24_xlmr_weight,

    "step24_semantic_weight":
        step24_semantic_weight,

    "emoji_threshold":
        best_emoji_threshold,

    "train_samples":
        len(train),

    "validation_samples":
        len(validation),

    "test_samples":
        len(test),

    "removed_train_validation":
        removed_train_validation,

    "removed_train_test":
        removed_train_test,

    "removed_validation_test":
        removed_validation_test,

    "leakage":
        {
            "train_validation":
                len(train_validation_overlap),

            "train_test":
                len(train_test_overlap),

            "validation_test":
                len(validation_test_overlap),
        },

    "baseline_metrics":
        baseline_metrics,

    "emoji_metrics":
        emoji_metrics,

    "metric_changes":
        {
            "accuracy":
                emoji_metrics["accuracy"]
                - baseline_metrics["accuracy"],

            "macro_f1":
                emoji_metrics["macro_f1"]
                - baseline_metrics["macro_f1"],

            "offensive_precision":
                emoji_metrics["offensive_precision"]
                - baseline_metrics["offensive_precision"],

            "offensive_recall":
                emoji_metrics["offensive_recall"]
                - baseline_metrics["offensive_recall"],

            "offensive_f1":
                emoji_metrics["offensive_f1"]
                - baseline_metrics["offensive_f1"],
        },
}

with open(
    os.path.join(
        OUT_DIR,
        "step31_config.json",
    ),
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        config,
        file,
        indent=2,
    )


# ==============================================================================
# 26. SAVE REPORT
# ==============================================================================

report_path = os.path.join(
    OUT_DIR,
    "step31_emoji_experiment_report.txt",
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:

    file.write("=" * 78 + "\n")
    file.write("STEP 31 — EMOJI-AWARE EXPERIMENT\n")
    file.write("=" * 78 + "\n\n")

    file.write("SEPARATE EXPERIMENTAL BRANCH\n\n")

    file.write("Step 20 retrained: NO\n")
    file.write("Step 24 retrained: NO\n")
    file.write("Step 26 retrained: NO\n")
    file.write("XLM-R recomputed: NO\n\n")

    file.write("DATA SPLITS\n")
    file.write("-" * 78 + "\n")
    file.write(f"Train: {len(train)}\n")
    file.write(f"Validation: {len(validation)}\n")
    file.write(f"Test: {len(test)}\n\n")

    file.write("DUPLICATE / LEAKAGE CLEANING\n")
    file.write("-" * 78 + "\n")
    file.write(
        f"Train-Validation removed: "
        f"{removed_train_validation}\n"
    )
    file.write(
        f"Train-Test removed: "
        f"{removed_train_test}\n"
    )
    file.write(
        f"Validation-Test removed: "
        f"{removed_validation_test}\n\n"
    )

    file.write("FINAL LEAKAGE CHECK\n")
    file.write("-" * 78 + "\n")
    file.write(
        f"Train-Validation overlap: "
        f"{len(train_validation_overlap)}\n"
    )
    file.write(
        f"Train-Test overlap: "
        f"{len(train_test_overlap)}\n"
    )
    file.write(
        f"Validation-Test overlap: "
        f"{len(validation_test_overlap)}\n\n"
    )

    file.write("FROZEN STEP 26 BASELINE\n")
    file.write("-" * 78 + "\n")
    for key, value in baseline_metrics.items():
        file.write(f"{key}: {value}\n")

    file.write("\nEMOJI EXPERIMENTAL BRANCH\n")
    file.write("-" * 78 + "\n")
    file.write(
        f"Threshold: {best_emoji_threshold}\n"
    )
    for key, value in emoji_metrics.items():
        file.write(f"{key}: {value}\n")

    file.write("\nMETRIC CHANGE: EMOJI - BASELINE\n")
    file.write("-" * 78 + "\n")
    for key in [
        "accuracy",
        "macro_f1",
        "offensive_precision",
        "offensive_recall",
        "offensive_f1",
    ]:
        delta = (
            emoji_metrics[key]
            - baseline_metrics[key]
        )
        file.write(
            f"{key}: {delta:+.6f}\n"
        )

    file.write("\nSTEP 26 ALIGNMENT REPORT\n")
    file.write("-" * 78 + "\n")
    file.write(
        alignment_report_df.to_string(
            index=False
        )
    )
    file.write("\n\n")

    file.write(
        "INTERPRETATION NOTE\n"
    )
    file.write("-" * 78 + "\n")
    file.write(
        "This is an emoji-only experimental branch. "
        "It does NOT prove that adding emoji features to the "
        "Step 24/Step 26 classifier improves that same model. "
        "It tests whether emoji-only features contain useful "
        "classification signal when evaluated against the frozen "
        "Step 26 independent-test baseline.\n"
    )


# ==============================================================================
# 27. FINAL SUMMARY
# ==============================================================================

section("STEP 31 COMPLETED")

print()
print("Frozen Step 26 baseline:")
print(
    f"  Accuracy : {baseline_metrics['accuracy']:.4f}"
)
print(
    f"  Macro-F1 : {baseline_metrics['macro_f1']:.4f}"
)
print(
    f"  Offensive F1 : "
    f"{baseline_metrics['offensive_f1']:.4f}"
)

print()
print("Emoji experimental branch:")
print(
    f"  Accuracy : {emoji_metrics['accuracy']:.4f}"
)
print(
    f"  Macro-F1 : {emoji_metrics['macro_f1']:.4f}"
)
print(
    f"  Offensive F1 : "
    f"{emoji_metrics['offensive_f1']:.4f}"
)

print()
print("Metric change (Emoji - Baseline):")
print(
    f"  Accuracy : "
    f"{emoji_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f}"
)
print(
    f"  Macro-F1 : "
    f"{emoji_metrics['macro_f1'] - baseline_metrics['macro_f1']:+.4f}"
)
print(
    f"  Offensive F1 : "
    f"{emoji_metrics['offensive_f1'] - baseline_metrics['offensive_f1']:+.4f}"
)

print()
print("Output folder:")
print(OUT_DIR)

print()
print("No Step 20 / Step 24 / Step 26 model was retrained or modified.")
print("STEP 31 FINISHED SUCCESSFULLY")
