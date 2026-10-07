"""
STEP 22 - MULTILINGUAL ERROR ANALYSIS

Uses Step 21 prediction output.

0 -> Non_Offensive
1 -> Offensive

No model training is performed.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = (
    "results/multilingual_model_comparison/"
    "predictions/combined_model_predictions.csv"
)

OUTPUT_DIR = "results/multilingual_error_analysis"

REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
ERROR_DIR = os.path.join(OUTPUT_DIR, "errors")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("STEP 22 - MULTILINGUAL ERROR ANALYSIS")
print("=" * 70)


# ================================================================
# 1. LOAD DATA
# ================================================================

print("=" * 70)
print("1. LOADING MODEL PREDICTIONS")
print("=" * 70)

print("Prediction file:")
print(INPUT_FILE)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Step 21 prediction file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nRecords: {len(df)}")
print("Columns:")
print(list(df.columns))


# ================================================================
# 2. VALIDATE COLUMNS
# ================================================================

print("=" * 70)
print("2. VALIDATING PREDICTION FORMAT")
print("=" * 70)

required_columns = [
    "true_label",
    "prediction",
    "language",
    "text",
    "model",
    "true_numeric",
    "prediction_numeric"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

print("Required columns found successfully.")


# ================================================================
# 3. MODEL IDENTIFICATION
# ================================================================

print("=" * 70)
print("3. IDENTIFYING MODELS")
print("=" * 70)

print("\nModels present:")

print(
    df["model"]
    .value_counts()
)


baseline_name = "Multilingual XLM-R Baseline"
minority_name = "Minority-Aware XLM-R"

if baseline_name not in df["model"].unique():
    raise ValueError(
        f"{baseline_name} not found."
    )

if minority_name not in df["model"].unique():
    raise ValueError(
        f"{minority_name} not found."
    )

baseline_df = df[
    df["model"] == baseline_name
].copy()

minority_df = df[
    df["model"] == minority_name
].copy()

print("\nBoth models found successfully.")

print("Baseline records:", len(baseline_df))
print("Minority-aware records:", len(minority_df))


# ================================================================
# 4. PREPARE NUMERIC LABELS
# ================================================================

print("=" * 70)
print("4. PREPARING NUMERIC LABELS")
print("=" * 70)

print("""
Label mapping:
0 -> Non_Offensive
1 -> Offensive
""")

for model_df in [baseline_df, minority_df]:

    model_df["true_numeric"] = pd.to_numeric(
        model_df["true_numeric"],
        errors="coerce"
    )

    model_df["prediction_numeric"] = pd.to_numeric(
        model_df["prediction_numeric"],
        errors="coerce"
    )

    model_df["correct"] = (
        model_df["true_numeric"]
        ==
        model_df["prediction_numeric"]
    )


# ================================================================
# 5. OVERALL ERROR ANALYSIS
# ================================================================

print("=" * 70)
print("5. OVERALL ERROR ANALYSIS")
print("=" * 70)

overall_rows = []

for model_name, model_df in [
    (baseline_name, baseline_df),
    (minority_name, minority_df)
]:

    total = len(model_df)

    correct = int(
        model_df["correct"].sum()
    )

    incorrect = total - correct

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    error_rate = (
        incorrect / total
        if total > 0
        else 0
    )

    overall_rows.append({
        "Model": model_name,
        "Total_Records": total,
        "Correct_Predictions": correct,
        "Incorrect_Predictions": incorrect,
        "Accuracy": accuracy,
        "Error_Rate": error_rate
    })


overall_errors = pd.DataFrame(
    overall_rows
)

print(
    overall_errors.to_string(
        index=False
    )
)

overall_errors.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "overall_error_analysis.csv"
    ),
    index=False
)


# ================================================================
# 6. LANGUAGE-WISE ERROR ANALYSIS
# ================================================================

print("=" * 70)
print("6. LANGUAGE-WISE ERROR ANALYSIS")
print("=" * 70)

language_rows = []

for model_name, model_df in [
    (baseline_name, baseline_df),
    (minority_name, minority_df)
]:

    for language in sorted(
        model_df["language"]
        .dropna()
        .unique()
    ):

        lang_df = model_df[
            model_df["language"] == language
        ]

        total = len(lang_df)

        correct = int(
            lang_df["correct"].sum()
        )

        incorrect = total - correct

        accuracy = (
            correct / total
            if total > 0
            else 0
        )

        error_rate = (
            incorrect / total
            if total > 0
            else 0
        )

        language_rows.append({
            "Model": model_name,
            "Language": language,
            "Total": total,
            "Correct": correct,
            "Incorrect": incorrect,
            "Accuracy": accuracy,
            "Error_Rate": error_rate
        })


language_errors = pd.DataFrame(
    language_rows
)

print(
    language_errors.to_string(
        index=False
    )
)

language_errors.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "language_wise_error_analysis.csv"
    ),
    index=False
)


# ================================================================
# 7. LANGUAGE ERROR COUNT
# ================================================================

print("=" * 70)
print("7. LANGUAGE-WISE ERROR COUNTS")
print("=" * 70)

language_error_counts = (
    language_errors[
        [
            "Language",
            "Model",
            "Incorrect"
        ]
    ]
    .pivot(
        index="Language",
        columns="Model",
        values="Incorrect"
    )
    .reset_index()
)

print(
    language_error_counts.to_string(
        index=False
    )
)

language_error_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "language_wise_error_counts.csv"
    ),
    index=False
)


# ================================================================
# 8. OFFENSIVE CLASS ANALYSIS
# ================================================================

print("=" * 70)
print("8. OFFENSIVE-CLASS ERROR ANALYSIS")
print("=" * 70)

offensive_rows = []

for model_name, model_df in [
    (baseline_name, baseline_df),
    (minority_name, minority_df)
]:

    # IMPORTANT:
    # Offensive = numeric label 1

    offensive_df = model_df[
        model_df["true_numeric"] == 1
    ]

    total_offensive = len(
        offensive_df
    )

    detected_offensive = offensive_df[
        offensive_df["prediction_numeric"] == 1
    ]

    missed_offensive = offensive_df[
        offensive_df["prediction_numeric"] != 1
    ]

    detected = len(
        detected_offensive
    )

    missed = len(
        missed_offensive
    )

    recall = (
        detected / total_offensive
        if total_offensive > 0
        else 0
    )

    offensive_rows.append({
        "Model": model_name,
        "True_Offensive": total_offensive,
        "Correctly_Detected_Offensive": detected,
        "Missed_Offensive": missed,
        "Offensive_Recall": recall
    })


offensive_analysis = pd.DataFrame(
    offensive_rows
)

print(
    offensive_analysis.to_string(
        index=False
    )
)

offensive_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "offensive_class_error_analysis.csv"
    ),
    index=False
)


# ================================================================
# 9. CONFUSION MATRICES
# ================================================================

print("=" * 70)
print("9. GENERATING CONFUSION MATRICES")
print("=" * 70)

labels = [0, 1]

label_names = [
    "Non_Offensive",
    "Offensive"
]

for model_name, model_df in [
    (baseline_name, baseline_df),
    (minority_name, minority_df)
]:

    cm = confusion_matrix(
        model_df["true_numeric"],
        model_df["prediction_numeric"],
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=label_names,
        columns=label_names
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    file_path = os.path.join(
        REPORT_DIR,
        f"{safe_name}_confusion_matrix.csv"
    )

    cm_df.to_csv(file_path)

    print(f"\n{model_name}")
    print(cm_df)


# ================================================================
# 10. CLASSIFICATION REPORT
# ================================================================

print("=" * 70)
print("10. GENERATING CLASSIFICATION REPORTS")
print("=" * 70)

for model_name, model_df in [
    (baseline_name, baseline_df),
    (minority_name, minority_df)
]:

    report = classification_report(
        model_df["true_numeric"],
        model_df["prediction_numeric"],
        labels=[0, 1],
        target_names=[
            "Non_Offensive",
            "Offensive"
        ],
        zero_division=0
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    file_path = os.path.join(
        REPORT_DIR,
        f"{safe_name}_classification_report.txt"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)

    print(f"\n{model_name}")
    print("-" * 50)
    print(report)


# ================================================================
# 11. SAVE INCORRECT PREDICTIONS
# ================================================================

print("=" * 70)
print("11. SAVING INCORRECT PREDICTIONS")
print("=" * 70)

baseline_incorrect = baseline_df[
    ~baseline_df["correct"]
].copy()

minority_incorrect = minority_df[
    ~minority_df["correct"]
].copy()

baseline_incorrect.to_csv(
    os.path.join(
        ERROR_DIR,
        "baseline_incorrect_predictions.csv"
    ),
    index=False
)

minority_incorrect.to_csv(
    os.path.join(
        ERROR_DIR,
        "minority_aware_incorrect_predictions.csv"
    ),
    index=False
)

print(
    "Baseline incorrect:",
    len(baseline_incorrect)
)

print(
    "Minority-aware incorrect:",
    len(minority_incorrect)
)


# ================================================================
# 12. LANGUAGE-SPECIFIC ERRORS
# ================================================================

print("=" * 70)
print("12. SAVING LANGUAGE-SPECIFIC ERRORS")
print("=" * 70)

languages = sorted(
    df["language"]
    .dropna()
    .unique()
)

for model_name, model_df, prefix in [
    (
        baseline_name,
        baseline_df,
        "baseline"
    ),
    (
        minority_name,
        minority_df,
        "minority_aware"
    )
]:

    incorrect = model_df[
        ~model_df["correct"]
    ]

    for language in languages:

        language_error_df = incorrect[
            incorrect["language"] == language
        ]

        filename = (
            f"{prefix}_"
            f"{str(language).lower()}_errors.csv"
        )

        language_error_df.to_csv(
            os.path.join(
                ERROR_DIR,
                filename
            ),
            index=False
        )

print(
    "Language-specific error files generated."
)


# ================================================================
# 13. REPRESENTATIVE ERRORS
# ================================================================

print("=" * 70)
print("13. SAVING REPRESENTATIVE ERROR EXAMPLES")
print("=" * 70)

for model_name, model_df in [
    (baseline_name, baseline_df),
    (minority_name, minority_df)
]:

    incorrect = model_df[
        ~model_df["correct"]
    ].copy()

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    sample = incorrect.head(100)

    sample.to_csv(
        os.path.join(
            ERROR_DIR,
            f"{safe_name}_representative_errors.csv"
        ),
        index=False
    )

    print(
        f"{model_name}: "
        f"{len(sample)} examples saved."
    )


# ================================================================
# 14. LANGUAGE-WISE ERROR PLOT
# ================================================================

print("=" * 70)
print("14. GENERATING LANGUAGE-WISE ERROR PLOT")
print("=" * 70)

plot_data = language_errors.pivot(
    index="Language",
    columns="Model",
    values="Error_Rate"
)

ax = plot_data.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Language-wise Error Rate Comparison"
)

ax.set_xlabel("Language")
ax.set_ylabel("Error Rate")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "language_wise_error_rate.png"
    ),
    dpi=300
)

plt.close()

print(
    "Language-wise error plot generated."
)


# ================================================================
# 15. OVERALL ERROR PLOT
# ================================================================

print("=" * 70)
print("15. GENERATING OVERALL ERROR PLOT")
print("=" * 70)

plt.figure(figsize=(9, 6))

plt.bar(
    overall_errors["Model"],
    overall_errors["Error_Rate"]
)

plt.title(
    "Overall Error Rate Comparison"
)

plt.xlabel("Model")
plt.ylabel("Error Rate")

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "overall_error_rate_comparison.png"
    ),
    dpi=300
)

plt.close()

print(
    "Overall error plot generated."
)


# ================================================================
# 16. FINAL SUMMARY
# ================================================================

print("=" * 70)
print("16. FINAL ERROR ANALYSIS SUMMARY")
print("=" * 70)

print("\nOVERALL ERROR ANALYSIS")
print(
    overall_errors.to_string(
        index=False
    )
)

print("\nLANGUAGE-WISE ERROR ANALYSIS")
print(
    language_errors.to_string(
        index=False
    )
)

print("\nOFFENSIVE-CLASS ANALYSIS")
print(
    offensive_analysis.to_string(
        index=False
    )
)


# ================================================================
# COMPLETION
# ================================================================

print("=" * 70)
print("STEP 22 - MULTILINGUAL ERROR ANALYSIS COMPLETED")
print("=" * 70)

print("\nAll results saved in:")
print(OUTPUT_DIR)

print("""
Generated files:

- overall_error_analysis.csv
- language_wise_error_analysis.csv
- language_wise_error_counts.csv
- offensive_class_error_analysis.csv

Reports:
- reports/*_classification_report.txt
- reports/*_confusion_matrix.csv

Errors:
- errors/baseline_incorrect_predictions.csv
- errors/minority_aware_incorrect_predictions.csv
- errors/*_language_errors.csv
- errors/*_representative_errors.csv

Plots:
- plots/language_wise_error_rate.png
- plots/overall_error_rate_comparison.png
""")

print("NO MODEL TRAINING WAS PERFORMED.")

print(
    "\nStep 22 identifies where the multilingual models "
    "make errors and evaluates offensive-class detection "
    "across Tamil, Malayalam and Kannada."
)

print("=" * 70)