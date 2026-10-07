import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

print("=" * 70)
print("STEP 13 - COMPREHENSIVE ERROR ANALYSIS")
print("=" * 70)

# ============================================================
# DIRECTORIES
# ============================================================

OUTPUT_DIR = "results/error_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_NAMES = {
    0: "Hate_Speech",
    1: "Neither",
    2: "Offensive_Language"
}

LABEL_TO_ID = {
    "Hate_Speech": 0,
    "Neither": 1,
    "Offensive_Language": 2
}

# ============================================================
# FILES
# ============================================================

MODEL_FILES = {
    "XLM-R Baseline":
        "results/transformer_evaluation/xlm_roberta_predictions.csv",

    "XLM-R Balanced":
        "results/balanced_transformer/balanced_xlm_roberta_predictions.csv",

    "XLM-R Weighted":
        "results/weighted_transformer/weighted_xlm_roberta_predictions.csv"
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def detect_text_column(df):
    candidates = [
        "model_text",
        "normalized_text",
        "tweet",
        "text"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def normalize_label(value):
    """
    Converts numeric IDs and string labels into
    a common textual label representation.
    """

    if pd.isna(value):
        return None

    # Numeric label
    if isinstance(value, (int, np.integer)):
        return LABEL_NAMES.get(int(value), str(value))

    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return LABEL_NAMES.get(int(value), str(value))

    value_str = str(value).strip()

    # Numeric string
    if value_str in ["0", "1", "2"]:
        return LABEL_NAMES[int(value_str)]

    # Already textual
    if value_str in LABEL_TO_ID:
        return value_str

    return value_str


def detect_actual_column(df, model_name):

    # Baseline XLM-R
    if model_name == "XLM-R Baseline":
        candidates = [
            "true_label",
            "actual_label",
            "label"
        ]

    else:
        candidates = [
            "actual_label",
            "true_label",
            "labels",
            "label"
        ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not detect actual label column for {model_name}"
    )


def detect_prediction_column(df):

    candidates = [
        "predicted_label",
        "prediction",
        "predicted_class"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not detect predicted label column."
    )


# ============================================================
# ANALYZE ONE MODEL
# ============================================================

def analyze_model(model_name, file_path):

    print()
    print("=" * 70)
    print(model_name.upper())
    print("=" * 70)

    print()
    print("Loading:")
    print(file_path)

    df = pd.read_csv(file_path)

    print(f"Loaded: {len(df)} records")

    print()
    print("Columns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # DETECT COLUMNS
    # --------------------------------------------------------

    actual_col = detect_actual_column(df, model_name)
    predicted_col = detect_prediction_column(df)
    text_col = detect_text_column(df)

    print()
    print("Detected actual label column:", actual_col)
    print("Detected predicted label column:", predicted_col)
    print("Detected text column:", text_col)

    # --------------------------------------------------------
    # NORMALIZE LABELS
    # --------------------------------------------------------

    df["_actual_normalized"] = df[actual_col].apply(
        normalize_label
    )

    df["_predicted_normalized"] = df[predicted_col].apply(
        normalize_label
    )

    # --------------------------------------------------------
    # VERIFY LABELS
    # --------------------------------------------------------

    print()
    print("Actual labels after normalization:")
    print(df["_actual_normalized"].value_counts())

    print()
    print("Predicted labels after normalization:")
    print(df["_predicted_normalized"].value_counts())

    # --------------------------------------------------------
    # CORRECT / INCORRECT
    # --------------------------------------------------------

    df["_correct"] = (
        df["_actual_normalized"]
        == df["_predicted_normalized"]
    )

    total = len(df)
    correct = int(df["_correct"].sum())
    incorrect = total - correct
    accuracy = correct / total

    print()
    print("Total samples :", total)
    print("Correct       :", correct)
    print("Incorrect     :", incorrect)
    print("Accuracy      :", f"{accuracy:.4f}")

    # --------------------------------------------------------
    # ERROR PAIRS
    # --------------------------------------------------------

    errors = df[~df["_correct"]].copy()

    if len(errors) > 0:

        errors["Error_Type"] = (
            errors["_actual_normalized"]
            + " -> "
            + errors["_predicted_normalized"]
        )

        error_pairs = (
            errors["Error_Type"]
            .value_counts()
            .reset_index()
        )

        error_pairs.columns = [
            "Error_Type",
            "Count"
        ]

    else:

        error_pairs = pd.DataFrame(
            columns=["Error_Type", "Count"]
        )

    print()
    print("Most common error pairs:")

    if len(error_pairs) > 0:
        print(error_pairs.head(10).to_string(index=False))
    else:
        print("No errors found.")

    # --------------------------------------------------------
    # SAVE MISCLASSIFIED SAMPLES
    # --------------------------------------------------------

    if len(errors) > 0:

        save_errors = errors.copy()

        drop_columns = [
            "_actual_normalized",
            "_predicted_normalized",
            "_correct"
        ]

        save_errors = save_errors.drop(
            columns=[
                col for col in drop_columns
                if col in save_errors.columns
            ]
        )

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("-", "")
        )

        error_file = (
            f"{OUTPUT_DIR}/"
            f"{safe_name}_misclassified.csv"
        )

        save_errors.to_csv(
            error_file,
            index=False
        )

        print()
        print("Misclassified samples saved:")
        print(error_file)

    # --------------------------------------------------------
    # SAVE ERROR PAIRS
    # --------------------------------------------------------

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    error_pair_file = (
        f"{OUTPUT_DIR}/"
        f"{safe_name}_error_pairs.csv"
    )

    error_pairs.to_csv(
        error_pair_file,
        index=False
    )

    print()
    print("Error-pair analysis saved:")
    print(error_pair_file)

    # --------------------------------------------------------
    # CLASS-WISE PERFORMANCE
    # --------------------------------------------------------

    class_rows = []

    for class_name in [
        "Hate_Speech",
        "Neither",
        "Offensive_Language"
    ]:

        actual_mask = (
            df["_actual_normalized"]
            == class_name
        )

        predicted_mask = (
            df["_predicted_normalized"]
            == class_name
        )

        actual_count = int(actual_mask.sum())
        predicted_count = int(predicted_mask.sum())

        correct_count = int(
            (
                actual_mask
                & predicted_mask
            ).sum()
        )

        # Precision
        if predicted_count > 0:
            precision = (
                correct_count / predicted_count
            )
        else:
            precision = 0.0

        # Recall
        if actual_count > 0:
            recall = (
                correct_count / actual_count
            )
        else:
            recall = 0.0

        # F1
        if precision + recall > 0:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        class_rows.append({
            "Model": model_name,
            "Class": class_name,
            "Actual": actual_count,
            "Predicted": predicted_count,
            "Correct": correct_count,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

    class_df = pd.DataFrame(class_rows)

    print()
    print("Class-wise performance:")
    print(class_df.to_string(index=False))

    class_file = (
        f"{OUTPUT_DIR}/"
        f"{safe_name}_class_performance.csv"
    )

    class_df.to_csv(
        class_file,
        index=False
    )

    print()
    print("Class-wise analysis saved:")
    print(class_file)

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {
        "Model": model_name,
        "Total": total,
        "Correct": correct,
        "Incorrect": incorrect,
        "Accuracy": accuracy,
        "Error_Rate": incorrect / total,
        "Class_Performance": class_df,
        "Error_Pairs": error_pairs
    }


# ============================================================
# RUN ALL MODELS
# ============================================================

results = {}

for model_name, file_path in MODEL_FILES.items():

    if not os.path.exists(file_path):

        print()
        print("WARNING: File not found:")
        print(file_path)

        continue

    results[model_name] = analyze_model(
        model_name,
        file_path
    )


# ============================================================
# HATE SPEECH PERFORMANCE COMPARISON
# ============================================================

print()
print("=" * 70)
print("HATE SPEECH PERFORMANCE COMPARISON")
print("=" * 70)

hate_rows = []

for model_name, result in results.items():

    class_df = result["Class_Performance"]

    hate = class_df[
        class_df["Class"] == "Hate_Speech"
    ]

    if len(hate) > 0:

        row = hate.iloc[0]

        hate_rows.append({
            "Model": model_name,
            "Actual Hate Speech": row["Actual"],
            "Predicted Hate Speech": row["Predicted"],
            "Correct": row["Correct"],
            "Precision": row["Precision"],
            "Recall": row["Recall"],
            "F1": row["F1"]
        })

hate_df = pd.DataFrame(hate_rows)

print(hate_df.to_string(index=False))

hate_file = (
    f"{OUTPUT_DIR}/hate_speech_comparison.csv"
)

hate_df.to_csv(
    hate_file,
    index=False
)

print()
print("Hate Speech comparison saved:")
print(hate_file)


# ============================================================
# ERROR COUNT COMPARISON
# ============================================================

print()
print("=" * 70)
print("ERROR COUNT COMPARISON")
print("=" * 70)

summary_rows = []

for model_name, result in results.items():

    summary_rows.append({
        "Model": model_name,
        "Total": result["Total"],
        "Correct": result["Correct"],
        "Incorrect": result["Incorrect"],
        "Accuracy": result["Accuracy"],
        "Error_Rate": result["Error_Rate"]
    })

summary_df = pd.DataFrame(summary_rows)

print(summary_df.to_string(index=False))

summary_file = (
    f"{OUTPUT_DIR}/error_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# HATE SPEECH RECALL PLOT
# ============================================================

print()
print("Generating Hate Speech recall comparison...")

if len(hate_df) > 0:

    plt.figure(figsize=(10, 6))

    plt.bar(
        hate_df["Model"],
        hate_df["Recall"]
    )

    plt.ylabel("Hate Speech Recall")
    plt.xlabel("Model")
    plt.title("Hate Speech Recall Comparison")
    plt.xticks(rotation=20)
    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/hate_speech_recall_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# ERROR RATE PLOT
# ============================================================

print("Generating error rate comparison...")

if len(summary_df) > 0:

    plt.figure(figsize=(10, 6))

    plt.bar(
        summary_df["Model"],
        summary_df["Error_Rate"]
    )

    plt.ylabel("Error Rate")
    plt.xlabel("Model")
    plt.title("Model Error Rate Comparison")
    plt.xticks(rotation=20)
    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/error_rate_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("STEP 13 - ERROR ANALYSIS COMPLETED")
print("=" * 70)

print()
print("Generated files:")
print("- results/error_analysis/xlm_r_baseline_misclassified.csv")
print("- results/error_analysis/xlm_r_balanced_misclassified.csv")
print("- results/error_analysis/xlm_r_weighted_misclassified.csv")
print("- results/error_analysis/xlm_r_baseline_error_pairs.csv")
print("- results/error_analysis/xlm_r_balanced_error_pairs.csv")
print("- results/error_analysis/xlm_r_weighted_error_pairs.csv")
print("- results/error_analysis/hate_speech_comparison.csv")
print("- results/error_analysis/error_summary.csv")
print("- results/error_analysis/hate_speech_recall_comparison.png")
print("- results/error_analysis/error_rate_comparison.png")

print()
print("=" * 70)
print("ERROR ANALYSIS FINISHED")
print("=" * 70)