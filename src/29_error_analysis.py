"""
================================================================================
STEP 29 - FINAL ERROR ANALYSIS
================================================================================

Purpose:
    Analyze actual errors made by the FINAL Step 24 system on the independent
    Step 26 test sets.

IMPORTANT:
    - NO MODEL TRAINING
    - NO MODEL RETRAINING
    - NO MODEL MODIFICATION
    - NO DATASET MODIFICATION
    - Uses ONLY saved Step 26 prediction CSV files
    - No manually invented examples

Analyzes:
    1. False Positives
    2. False Negatives
    3. Error counts and rates
    4. Language-wise errors
    5. High-confidence false positives
    6. High-confidence false negatives
    7. Error examples
    8. Overall error statistics

Outputs:
    results\multilingual_cross_dataset_final\step29_error_analysis\

================================================================================
"""

import os
import pandas as pd
import numpy as np


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = r"E:\Offensive Language Detection in Social Media_NLP"

STEP26_DIR = os.path.join(
    ROOT,
    "results",
    "multilingual_cross_dataset_final",
    "step24_semantic_final",
    "final_evaluation"
)

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "multilingual_cross_dataset_final",
    "step29_error_analysis"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# =============================================================================
# 2. INPUT FILES
# =============================================================================

FILES = {
    "English": os.path.join(
        STEP26_DIR,
        "dataset1_english_test_predictions.csv"
    ),

    "Tamil": os.path.join(
        STEP26_DIR,
        "dataset2_tamil_test_predictions.csv"
    ),

    "Malayalam": os.path.join(
        STEP26_DIR,
        "dataset2_malayalam_test_predictions.csv"
    ),

    "Kannada": os.path.join(
        STEP26_DIR,
        "dataset2_kannada_test_predictions.csv"
    ),

    "Dravidian_Combined": os.path.join(
        STEP26_DIR,
        "dataset2_dravidian_combined_test_predictions.csv"
    ),

    "Overall": os.path.join(
        STEP26_DIR,
        "overall_independent_test_predictions.csv"
    )
}


# =============================================================================
# 3. CHECK FILES
# =============================================================================

print("=" * 80)
print("STEP 29 - FINAL ERROR ANALYSIS")
print("=" * 80)

for name, path in FILES.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired prediction file not found:\n"
            f"{name}\n"
            f"{path}"
        )

    print(
        f"[FOUND] {name}"
    )


print("\nAll Step 26 prediction files found.")


# =============================================================================
# 4. LABEL NORMALIZATION
# =============================================================================

def normalize_label(value):

    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (int, np.integer)
    ):

        if value in [0, 1]:
            return int(value)

    if isinstance(
        value,
        (float, np.floating)
    ):

        if value in [0.0, 1.0]:
            return int(value)

    text = str(
        value
    ).strip().lower()

    if text in [
        "0",
        "0.0",
        "non_offensive",
        "non-offensive",
        "non offensive",
        "not offensive",
        "not_offensive",
        "not-offensive"
    ]:

        return 0

    if text in [
        "1",
        "1.0",
        "offensive"
    ]:

        return 1

    return np.nan


# =============================================================================
# 5. LOAD AND PREPARE
# =============================================================================

def load_predictions(
    language,
    path
):

    print(
        f"\nLoading {language} predictions..."
    )

    df = pd.read_csv(
        path
    )

    required_columns = [
        "text",
        "predicted_label"
    ]

    for col in required_columns:

        if col not in df.columns:

            raise ValueError(
                f"{col} missing from {language} prediction file."
            )

    # -------------------------------------------------------------
    # Actual label
    # -------------------------------------------------------------

    if "label_id" in df.columns:

        df["actual_label"] = df[
            "label_id"
        ].apply(
            normalize_label
        )

    elif "label" in df.columns:

        df["actual_label"] = df[
            "label"
        ].apply(
            normalize_label
        )

    else:

        raise ValueError(
            f"Could not find actual label in {language}."
        )

    # -------------------------------------------------------------
    # Predicted label
    # -------------------------------------------------------------

    df["predicted_label_clean"] = df[
        "predicted_label"
    ].apply(
        normalize_label
    )

    # -------------------------------------------------------------
    # Probabilities
    # -------------------------------------------------------------

    if "hybrid_probability" in df.columns:

        df["confidence"] = np.where(
            df["predicted_label_clean"] == 1,
            df["hybrid_probability"],
            1.0 - df["hybrid_probability"]
        )

    else:

        df["confidence"] = np.nan

    # -------------------------------------------------------------
    # Error type
    # -------------------------------------------------------------

    df["error_type"] = "Correct"

    df.loc[
        (
            df["actual_label"] == 0
        )
        &
        (
            df["predicted_label_clean"] == 1
        ),
        "error_type"
    ] = "False Positive"

    df.loc[
        (
            df["actual_label"] == 1
        )
        &
        (
            df["predicted_label_clean"] == 0
        ),
        "error_type"
    ] = "False Negative"

    # -------------------------------------------------------------
    # Language
    # -------------------------------------------------------------

    df["analysis_language"] = language

    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    valid = (
        df["actual_label"].notna()
        &
        df["predicted_label_clean"].notna()
    )

    df = df[
        valid
    ].copy()

    df["actual_label"] = df[
        "actual_label"
    ].astype(int)

    df["predicted_label_clean"] = df[
        "predicted_label_clean"
    ].astype(int)

    print(
        f"Valid predictions: {len(df)}"
    )

    return df


# =============================================================================
# 6. LOAD LANGUAGE DATASETS
# =============================================================================

english = load_predictions(
    "English",
    FILES["English"]
)

tamil = load_predictions(
    "Tamil",
    FILES["Tamil"]
)

malayalam = load_predictions(
    "Malayalam",
    FILES["Malayalam"]
)

kannada = load_predictions(
    "Kannada",
    FILES["Kannada"]
)


# =============================================================================
# 7. COMBINE LANGUAGE DATA
# =============================================================================

language_df = pd.concat(
    [
        english,
        tamil,
        malayalam,
        kannada
    ],
    ignore_index=True
)


# =============================================================================
# 8. ERROR SUMMARY FUNCTION
# =============================================================================

def calculate_error_summary(
    df,
    dataset_name
):

    total = len(df)

    fp = int(
        (
            df["error_type"]
            ==
            "False Positive"
        ).sum()
    )

    fn = int(
        (
            df["error_type"]
            ==
            "False Negative"
        ).sum()
    )

    correct = int(
        (
            df["error_type"]
            ==
            "Correct"
        ).sum()
    )

    total_errors = fp + fn

    error_rate = (
        total_errors / total
        if total > 0
        else 0
    )

    return {
        "Dataset / Language": dataset_name,
        "Total Samples": total,
        "Correct": correct,
        "False Positives": fp,
        "False Negatives": fn,
        "Total Errors": total_errors,
        "Error Rate": error_rate
    }


# =============================================================================
# 9. LANGUAGE ERROR SUMMARY
# =============================================================================

summary_rows = []

summary_rows.append(
    calculate_error_summary(
        english,
        "English"
    )
)

summary_rows.append(
    calculate_error_summary(
        tamil,
        "Tamil"
    )
)

summary_rows.append(
    calculate_error_summary(
        malayalam,
        "Malayalam"
    )
)

summary_rows.append(
    calculate_error_summary(
        kannada,
        "Kannada"
    )
)

summary_rows.append(
    calculate_error_summary(
        language_df,
        "All Languages"
    )
)


error_summary = pd.DataFrame(
    summary_rows
)


# =============================================================================
# 10. PRINT ERROR SUMMARY
# =============================================================================

print("\n")
print("=" * 80)
print("ERROR SUMMARY")
print("=" * 80)

print(
    error_summary.to_string(
        index=False
    )
)


# =============================================================================
# 11. SAVE ERROR SUMMARY
# =============================================================================

error_summary_path = os.path.join(
    OUT_DIR,
    "error_summary.csv"
)

error_summary.to_csv(
    error_summary_path,
    index=False
)

print(
    "\nSaved:",
    error_summary_path
)


# =============================================================================
# 12. FALSE POSITIVE / FALSE NEGATIVE TABLES
# =============================================================================

false_positives = language_df[
    language_df["error_type"]
    ==
    "False Positive"
].copy()

false_negatives = language_df[
    language_df["error_type"]
    ==
    "False Negative"
].copy()


# =============================================================================
# 13. SORT BY CONFIDENCE
# =============================================================================

false_positives = false_positives.sort_values(
    "confidence",
    ascending=False
)

false_negatives = false_negatives.sort_values(
    "confidence",
    ascending=False
)


# =============================================================================
# 14. SAVE ALL FALSE POSITIVES
# =============================================================================

fp_columns = [
    "analysis_language",
    "text",
    "actual_label",
    "predicted_label_clean",
    "hybrid_probability",
    "xlmr_probability",
    "semantic_probability",
    "confidence",
    "predicted_name"
]

fp_columns = [
    col
    for col in fp_columns
    if col in false_positives.columns
]


fp_path = os.path.join(
    OUT_DIR,
    "all_false_positives.csv"
)

false_positives[
    fp_columns
].to_csv(
    fp_path,
    index=False
)

print(
    "Saved:",
    fp_path
)


# =============================================================================
# 15. SAVE ALL FALSE NEGATIVES
# =============================================================================

fn_columns = [
    "analysis_language",
    "text",
    "actual_label",
    "predicted_label_clean",
    "hybrid_probability",
    "xlmr_probability",
    "semantic_probability",
    "confidence",
    "predicted_name"
]

fn_columns = [
    col
    for col in fn_columns
    if col in false_negatives.columns
]


fn_path = os.path.join(
    OUT_DIR,
    "all_false_negatives.csv"
)

false_negatives[
    fn_columns
].to_csv(
    fn_path,
    index=False
)

print(
    "Saved:",
    fn_path
)


# =============================================================================
# 16. LANGUAGE-WISE FALSE POSITIVES
# =============================================================================

fp_language = (
    false_positives
    .groupby("analysis_language")
    .size()
    .reset_index(
        name="False Positives"
    )
)

fp_language_path = os.path.join(
    OUT_DIR,
    "false_positives_by_language.csv"
)

fp_language.to_csv(
    fp_language_path,
    index=False
)


# =============================================================================
# 17. LANGUAGE-WISE FALSE NEGATIVES
# =============================================================================

fn_language = (
    false_negatives
    .groupby("analysis_language")
    .size()
    .reset_index(
        name="False Negatives"
    )
)

fn_language_path = os.path.join(
    OUT_DIR,
    "false_negatives_by_language.csv"
)

fn_language.to_csv(
    fn_language_path,
    index=False
)


# =============================================================================
# 18. HIGH-CONFIDENCE FALSE POSITIVES
# =============================================================================
#
# These are cases where the model was confident that the text was offensive,
# but the true label is Non-Offensive.
# =============================================================================

high_conf_fp = false_positives[
    false_positives["confidence"] >= 0.80
].copy()

high_conf_fp_path = os.path.join(
    OUT_DIR,
    "high_confidence_false_positives.csv"
)

high_conf_fp[
    fp_columns
].to_csv(
    high_conf_fp_path,
    index=False
)


# =============================================================================
# 19. HIGH-CONFIDENCE FALSE NEGATIVES
# =============================================================================
#
# These are cases where the model was confident that the text was non-offensive,
# but the true label is Offensive.
# =============================================================================

high_conf_fn = false_negatives[
    false_negatives["confidence"] >= 0.80
].copy()

high_conf_fn_path = os.path.join(
    OUT_DIR,
    "high_confidence_false_negatives.csv"
)

high_conf_fn[
    fn_columns
].to_csv(
    high_conf_fn_path,
    index=False
)


# =============================================================================
# 20. TOP 20 FALSE POSITIVES
# =============================================================================

top_fp = false_positives.head(
    20
)

top_fp_path = os.path.join(
    OUT_DIR,
    "top20_false_positives.csv"
)

top_fp[
    fp_columns
].to_csv(
    top_fp_path,
    index=False
)


# =============================================================================
# 21. TOP 20 FALSE NEGATIVES
# =============================================================================

top_fn = false_negatives.head(
    20
)

top_fn_path = os.path.join(
    OUT_DIR,
    "top20_false_negatives.csv"
)

top_fn[
    fn_columns
].to_csv(
    top_fn_path,
    index=False
)


# =============================================================================
# 22. ERROR ANALYSIS TEXT REPORT
# =============================================================================

report_path = os.path.join(
    OUT_DIR,
    "error_analysis_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 80 + "\n")
    f.write("STEP 29 - FINAL ERROR ANALYSIS\n")
    f.write("=" * 80 + "\n\n")

    f.write(
        "Analysis source: Step 26 independent-test predictions.\n"
    )

    f.write(
        "No model was trained or modified.\n"
    )

    f.write(
        "No manually invented examples are included.\n\n"
    )

    f.write(
        "ERROR SUMMARY\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        error_summary.to_string(
            index=False
        )
    )

    f.write("\n\n")

    # -------------------------------------------------------------
    # Overall interpretation
    # -------------------------------------------------------------

    total = len(language_df)

    total_fp = len(false_positives)

    total_fn = len(false_negatives)

    total_errors = total_fp + total_fn

    f.write(
        "OVERALL ERROR COUNTS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        f"Total samples : {total}\n"
    )

    f.write(
        f"Correct       : {total - total_errors}\n"
    )

    f.write(
        f"False Positives: {total_fp}\n"
    )

    f.write(
        f"False Negatives: {total_fn}\n"
    )

    f.write(
        f"Total errors  : {total_errors}\n"
    )

    f.write(
        f"Error rate    : {total_errors / total:.4f}\n\n"
    )

    # -------------------------------------------------------------
    # Language interpretation
    # -------------------------------------------------------------

    f.write(
        "LANGUAGE-WISE ERROR ANALYSIS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    for _, row in error_summary.iterrows():

        f.write(
            f"\n{row['Dataset / Language']}\n"
        )

        f.write(
            f"  Samples       : {int(row['Total Samples'])}\n"
        )

        f.write(
            f"  Correct       : {int(row['Correct'])}\n"
        )

        f.write(
            f"  False Positive: {int(row['False Positives'])}\n"
        )

        f.write(
            f"  False Negative: {int(row['False Negatives'])}\n"
        )

        f.write(
            f"  Error Rate    : "
            f"{float(row['Error Rate']):.4f}\n"
        )

    # -------------------------------------------------------------
    # High confidence errors
    # -------------------------------------------------------------

    f.write(
        "\n\nHIGH-CONFIDENCE ERROR COUNTS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        f"High-confidence False Positives (>= 0.80): "
        f"{len(high_conf_fp)}\n"
    )

    f.write(
        f"High-confidence False Negatives (>= 0.80): "
        f"{len(high_conf_fn)}\n"
    )

    # -------------------------------------------------------------
    # Actual examples
    # -------------------------------------------------------------

    f.write(
        "\n\nTOP FALSE POSITIVE EXAMPLES\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    for idx, (_, row) in enumerate(
        top_fp.iterrows(),
        start=1
    ):

        f.write(
            f"\n{idx}. [{row['analysis_language']}]\n"
        )

        f.write(
            f"Text: {row['text']}\n"
        )

        f.write(
            f"True label: Non-Offensive\n"
        )

        f.write(
            f"Predicted: Offensive\n"
        )

        if "hybrid_probability" in row:

            f.write(
                f"Hybrid probability: "
                f"{row['hybrid_probability']:.4f}\n"
            )

    f.write(
        "\n\nTOP FALSE NEGATIVE EXAMPLES\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    for idx, (_, row) in enumerate(
        top_fn.iterrows(),
        start=1
    ):

        f.write(
            f"\n{idx}. [{row['analysis_language']}]\n"
        )

        f.write(
            f"Text: {row['text']}\n"
        )

        f.write(
            f"True label: Offensive\n"
        )

        f.write(
            f"Predicted: Non-Offensive\n"
        )

        if "hybrid_probability" in row:

            f.write(
                f"Hybrid probability: "
                f"{row['hybrid_probability']:.4f}\n"
            )


# =============================================================================
# 23. FINAL PRINT
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 29 COMPLETE")
print("=" * 80)

print("\nError summary:")
print(
    error_summary.to_string(
        index=False
    )
)

print("\nTotal False Positives :", len(false_positives))
print("Total False Negatives :", len(false_negatives))
print(
    "Total Errors          :",
    len(false_positives) + len(false_negatives)
)

print(
    "\nHigh-confidence False Positives (>=0.80):",
    len(high_conf_fp)
)

print(
    "High-confidence False Negatives (>=0.80):",
    len(high_conf_fn)
)

print("\nOutput directory:")
print(OUT_DIR)

print("\nImportant output files:")
print(" - error_summary.csv")
print(" - all_false_positives.csv")
print(" - all_false_negatives.csv")
print(" - high_confidence_false_positives.csv")
print(" - high_confidence_false_negatives.csv")
print(" - top20_false_positives.csv")
print(" - top20_false_negatives.csv")
print(" - error_analysis_report.txt")
