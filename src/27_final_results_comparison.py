"""
================================================================================
STEP 27 - FINAL RESULTS COMPARISON
================================================================================

Purpose:
    Create the final results table for the project using ALREADY COMPLETED
    experiments.

IMPORTANT:
    - NO model training
    - NO model retraining
    - NO changes to Step 20
    - NO changes to Step 24
    - NO changes to Step 26

Outputs:
    results\multilingual_cross_dataset_final\step27_final_results\
        final_results_comparison.csv
        final_results_comparison.txt
        language_wise_results.csv
        project_final_metrics.json

================================================================================
"""

import os
import json
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = r"E:\Offensive Language Detection in Social Media_NLP"

RESULTS_ROOT = os.path.join(
    ROOT,
    "results",
    "multilingual_cross_dataset_final"
)

STEP24_DIR = os.path.join(
    RESULTS_ROOT,
    "step24_semantic_final"
)

STEP26_DIR = os.path.join(
    STEP24_DIR,
    "final_evaluation"
)

OUT_DIR = os.path.join(
    RESULTS_ROOT,
    "step27_final_results"
)

os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# 2. CHECK REQUIRED STEP 26 FILE
# =============================================================================

STEP26_SUMMARY = os.path.join(
    STEP26_DIR,
    "final_independent_test_summary.csv"
)

if not os.path.exists(STEP26_SUMMARY):
    raise FileNotFoundError(
        "\nStep 26 final summary was not found:\n"
        f"{STEP26_SUMMARY}\n\n"
        "Run Step 26 first."
    )


# =============================================================================
# 3. LOAD STEP 26 FINAL RESULTS
# =============================================================================

print("=" * 80)
print("STEP 27 - FINAL RESULTS COMPARISON")
print("=" * 80)

print("\nLoading Step 26 final independent-test results...")

step26 = pd.read_csv(STEP26_SUMMARY)

print("\nStep 26 results loaded successfully.")
print(step26.to_string(index=False))


# =============================================================================
# 4. STEP 24 VALIDATION RESULTS
# =============================================================================
#
# These are the validation/calibration results already obtained in Step 24.
# They are NOT final independent-test results.
#
# Step 24:
# Accuracy       = 0.8075
# Macro-F1       = 0.7399
# Offensive P    = 0.6100
# Offensive R    = 0.6046
# Offensive F1   = 0.6073
#
# Calibration:
# Threshold      = 0.65
# XLM-R weight   = 0.55
# Semantic weight= 0.45
# =============================================================================

STEP24_VALIDATION = {
    "Experiment": "Step 24 - Validation",
    "Dataset": "Combined Validation",
    "Samples": None,
    "Accuracy": 0.8075,
    "Macro-F1": 0.7399,
    "Offensive Precision": 0.6100,
    "Offensive Recall": 0.6046,
    "Offensive F1": 0.6073
}


# =============================================================================
# 5. STEP 20 BASELINE
# =============================================================================
#
# IMPORTANT:
# Enter the EXACT Step 20 FINAL TEST/RESULT values that you already obtained.
#
# DO NOT GUESS.
#
# If you have not finalized the Step 20 numbers yet, leave the values as
# None. The script will still generate the Step 24 and Step 26 results.
# =============================================================================

STEP20_BASELINE = {
    "Experiment": "Step 20 - Frozen XLM-R Baseline",
    "Dataset": "Cross-Dataset Evaluation",
    "Samples": None,

    # ============================================================
    # REPLACE THESE ONLY IF YOU HAVE THE EXACT STEP 20 RESULTS
    # ============================================================

    "Accuracy": None,
    "Macro-F1": None,
    "Offensive Precision": None,
    "Offensive Recall": None,
    "Offensive F1": None
}


# =============================================================================
# 6. CREATE MAIN COMPARISON TABLE
# =============================================================================

comparison_rows = [
    STEP20_BASELINE,
    STEP24_VALIDATION
]


# Add Step 26 rows
for _, row in step26.iterrows():

    comparison_rows.append({
        "Experiment": "Step 26 - Final Independent Test",
        "Dataset": row["Dataset / Language"],
        "Samples": int(row["Samples"]),
        "Accuracy": float(row["Accuracy"]),
        "Macro-F1": float(row["Macro-F1"]),
        "Offensive Precision": float(row["Offensive Precision"]),
        "Offensive Recall": float(row["Offensive Recall"]),
        "Offensive F1": float(row["Offensive F1"])
    })


comparison_df = pd.DataFrame(comparison_rows)


# =============================================================================
# 7. SAVE MAIN COMPARISON CSV
# =============================================================================

comparison_csv = os.path.join(
    OUT_DIR,
    "final_results_comparison.csv"
)

comparison_df.to_csv(
    comparison_csv,
    index=False
)


# =============================================================================
# 8. LANGUAGE-WISE TABLE
# =============================================================================

language_rows = []

for language in [
    "DATASET 1 - ENGLISH TEST",
    "DATASET 2 - Tamil TEST",
    "DATASET 2 - Malayalam TEST",
    "DATASET 2 - Kannada TEST",
    "DATASET 2 - DRAVIDIAN COMBINED TEST",
    "OVERALL INDEPENDENT TEST"
]:

    match = step26[
        step26["Dataset / Language"].str.strip() == language
    ]

    if len(match) == 1:

        row = match.iloc[0]

        language_rows.append({
            "Dataset / Language": language,
            "Samples": int(row["Samples"]),
            "Accuracy": float(row["Accuracy"]),
            "Macro-F1": float(row["Macro-F1"]),
            "Offensive Precision": float(row["Offensive Precision"]),
            "Offensive Recall": float(row["Offensive Recall"]),
            "Offensive F1": float(row["Offensive F1"])
        })


language_df = pd.DataFrame(language_rows)

language_csv = os.path.join(
    OUT_DIR,
    "language_wise_results.csv"
)

language_df.to_csv(
    language_csv,
    index=False
)


# =============================================================================
# 9. FINAL SYSTEM CONFIGURATION
# =============================================================================

final_configuration = {
    "Frozen Step 20 XLM-R": True,
    "Step 24 Semantic Classifier Retrained in Step 27": False,
    "Step 26 Retrained in Step 27": False,
    "XLM-R Weight": 0.55,
    "Semantic Weight": 0.45,
    "Decision Threshold": 0.65,
    "Mean + Max Contextual Representation": True,
    "Keyword Rules": False,
    "Manual Sentence Rules": False,
    "Offensive Vocabulary Hardcoded": False
}


config_json = os.path.join(
    OUT_DIR,
    "project_final_metrics.json"
)

with open(
    config_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_configuration,
        f,
        indent=4
    )


# =============================================================================
# 10. CREATE HUMAN-READABLE REPORT
# =============================================================================

txt_file = os.path.join(
    OUT_DIR,
    "final_results_comparison.txt"
)

with open(
    txt_file,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 90 + "\n")
    f.write("FINAL RESULTS COMPARISON - OFFENSIVE LANGUAGE DETECTION\n")
    f.write("=" * 90 + "\n\n")

    f.write(
        "Step 27 uses already completed experimental results.\n"
        "No model is trained or retrained in this step.\n\n"
    )

    f.write("-" * 90 + "\n")
    f.write("MAIN RESULTS\n")
    f.write("-" * 90 + "\n\n")

    f.write(
        comparison_df.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write("-" * 90 + "\n")
    f.write("LANGUAGE-WISE FINAL INDEPENDENT TEST RESULTS\n")
    f.write("-" * 90 + "\n\n")

    f.write(
        language_df.to_string(
            index=False
        )
    )

    f.write("\n\n")
    f.write("-" * 90 + "\n")
    f.write("FINAL SYSTEM CONFIGURATION\n")
    f.write("-" * 90 + "\n\n")

    for key, value in final_configuration.items():

        f.write(
            f"{key}: {value}\n"
        )


# =============================================================================
# 11. PRINT FINAL OUTPUT
# =============================================================================

print("\n")
print("=" * 80)
print("FINAL RESULTS COMPARISON")
print("=" * 80)

print(
    comparison_df.to_string(
        index=False
    )
)

print("\n")
print("=" * 80)
print("LANGUAGE-WISE FINAL INDEPENDENT TEST RESULTS")
print("=" * 80)

print(
    language_df.to_string(
        index=False
    )
)


# =============================================================================
# 12. FINAL OVERALL RESULT
# =============================================================================

overall = step26[
    step26["Dataset / Language"].str.strip()
    == "OVERALL INDEPENDENT TEST"
]

if len(overall) == 1:

    row = overall.iloc[0]

    print("\n")
    print("=" * 80)
    print("FINAL PROJECT RESULT")
    print("=" * 80)

    print(
        f"Samples              : {int(row['Samples'])}"
    )

    print(
        f"Accuracy             : {row['Accuracy']:.4f}"
    )

    print(
        f"Macro-F1             : {row['Macro-F1']:.4f}"
    )

    print(
        f"Offensive Precision  : {row['Offensive Precision']:.4f}"
    )

    print(
        f"Offensive Recall     : {row['Offensive Recall']:.4f}"
    )

    print(
        f"Offensive F1         : {row['Offensive F1']:.4f}"
    )


# =============================================================================
# 13. OUTPUT FILES
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 27 COMPLETE")
print("=" * 80)

print("\nCreated files:")

print(
    f"1. {comparison_csv}"
)

print(
    f"2. {language_csv}"
)

print(
    f"3. {txt_file}"
)

print(
    f"4. {config_json}"
)

print("\nNo model was trained or modified.")
print("No Step 20/24/26 rerun is required.")