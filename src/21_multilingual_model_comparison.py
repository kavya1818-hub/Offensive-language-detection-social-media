"""
STEP 21 - MULTILINGUAL MODEL COMPARISON

Purpose:
1. Load Step 19 Multilingual XLM-R baseline results
2. Load Step 20 Minority-Aware XLM-R results
3. Automatically locate prediction CSV files
4. Compare overall performance
5. Compare language-wise performance
6. Compare minority/offensive-class performance
7. Generate comparison tables and plots
8. Save all results under:
   results/multilingual_model_comparison/

No model training is performed.
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

BASELINE_DIR = "results/multilingual_baseline"
MINORITY_DIR = "results/multilingual_minority_aware"

OUTPUT_DIR = "results/multilingual_model_comparison"
PREDICTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "predictions")
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def find_prediction_csv(folder):
    """
    Search recursively for CSV files that appear to contain predictions.
    """

    if not os.path.exists(folder):
        return None

    csv_files = glob.glob(
        os.path.join(folder, "**", "*.csv"),
        recursive=True
    )

    if not csv_files:
        return None

    # Strong keywords first
    priority_files = []

    for file in csv_files:
        name = os.path.basename(file).lower()

        if (
            "prediction" in name
            or "predictions" in name
            or "test_prediction" in name
            or "test_predictions" in name
        ):
            priority_files.append(file)

    # Test candidates
    candidates = priority_files if priority_files else csv_files

    for file in candidates:

        try:
            df = pd.read_csv(file)

            columns_lower = {
                str(c).lower(): c
                for c in df.columns
            }

            has_prediction = any(
                key in columns_lower
                for key in [
                    "prediction",
                    "predicted_label",
                    "predicted",
                    "pred"
                ]
            )

            has_actual = any(
                key in columns_lower
                for key in [
                    "label",
                    "true_label",
                    "actual_label",
                    "mapped_label",
                    "true"
                ]
            )

            if has_prediction and has_actual:
                return file

        except Exception:
            continue

    return None


def find_column(df, possible_names):
    """
    Find a column using case-insensitive matching.
    """

    lower_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in possible_names:

        if name.lower() in lower_map:
            return lower_map[name.lower()]

    # Partial matching
    for col in df.columns:

        col_lower = str(col).lower()

        for name in possible_names:

            if name.lower() in col_lower:
                return col

    return None


def load_prediction_file(folder, model_name):

    print()
    print(f"Searching for {model_name} prediction file:")
    print(folder)

    prediction_file = find_prediction_csv(folder)

    if prediction_file is None:

        print()
        print(f"ERROR: Could not find prediction CSV for {model_name}.")
        print("Searching directory:")
        print(os.path.abspath(folder))

        print()
        print("CSV files found:")

        csv_files = glob.glob(
            os.path.join(folder, "**", "*.csv"),
            recursive=True
        )

        if csv_files:

            for file in csv_files:
                print(" -", file)

        else:
            print("No CSV files found.")

        return None

    print("Prediction file found:")
    print(prediction_file)

    df = pd.read_csv(prediction_file)

    print("Records:", len(df))
    print("Columns:", list(df.columns))

    return df


def normalize_label(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    lower = value.lower()

    if lower in [
        "1",
        "offensive",
        "off",
        "offensive_language"
    ]:
        return 1

    if lower in [
        "0",
        "non_offensive",
        "non-offensive",
        "not_offensive",
        "not offensive",
        "non offensive"
    ]:
        return 0

    return value


def prepare_predictions(df, model_name):

    print_header(f"PREPARING {model_name.upper()} PREDICTIONS")

    actual_col = find_column(
        df,
        [
            "mapped_label",
            "true_label",
            "actual_label",
            "label",
            "true"
        ]
    )

    prediction_col = find_column(
        df,
        [
            "prediction",
            "predicted_label",
            "predicted",
            "pred"
        ]
    )

    language_col = find_column(
        df,
        [
            "language",
            "lang"
        ]
    )

    text_col = find_column(
        df,
        [
            "text",
            "sentence",
            "comment"
        ]
    )

    if actual_col is None:
        raise ValueError(
            f"Could not identify actual-label column for {model_name}."
        )

    if prediction_col is None:
        raise ValueError(
            f"Could not identify prediction column for {model_name}."
        )

    print("Actual label column    :", actual_col)
    print("Prediction column      :", prediction_col)
    print("Language column        :", language_col)

    result = pd.DataFrame()

    result["true_label"] = df[actual_col].apply(normalize_label)
    result["prediction"] = df[prediction_col].apply(normalize_label)

    if language_col is not None:
        result["language"] = df[language_col].astype(str)
    else:
        result["language"] = "Unknown"

    if text_col is not None:
        result["text"] = df[text_col].astype(str)

    result["model"] = model_name

    # Remove invalid records
    result = result.dropna(
        subset=["true_label", "prediction"]
    ).reset_index(drop=True)

    return result


def convert_binary_labels(df):

    mapping = {
        "0": 0,
        "1": 1,
        "Non_Offensive": 0,
        "Offensive": 1,
        "non_offensive": 0,
        "offensive": 1,
        "Non-Offensive": 0,
        "non-offensive": 0
    }

    def convert(value):

        if isinstance(value, (int, np.integer)):
            return int(value)

        if isinstance(value, float):
            return int(value)

        value_str = str(value).strip()

        if value_str in mapping:
            return mapping[value_str]

        lower = value_str.lower()

        if lower == "offensive":
            return 1

        if lower in [
            "non_offensive",
            "non-offensive",
            "not_offensive",
            "not offensive",
            "non offensive"
        ]:
            return 0

        try:
            return int(float(value_str))
        except Exception:
            return np.nan

    df["true_numeric"] = df["true_label"].apply(convert)
    df["prediction_numeric"] = df["prediction"].apply(convert)

    df = df.dropna(
        subset=[
            "true_numeric",
            "prediction_numeric"
        ]
    ).copy()

    df["true_numeric"] = df["true_numeric"].astype(int)
    df["prediction_numeric"] = df["prediction_numeric"].astype(int)

    return df


def calculate_metrics(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1_Score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }


# ================================================================
# STEP 1 - LOAD BASELINE
# ================================================================

print_header("STEP 21 - MULTILINGUAL MODEL COMPARISON")

print_header("1. LOADING STEP 19 - MULTILINGUAL XLM-R BASELINE")

baseline_df = load_prediction_file(
    BASELINE_DIR,
    "Multilingual XLM-R Baseline"
)

if baseline_df is None:
    raise FileNotFoundError(
        "\nStep 19 prediction CSV was not found.\n"
        "Check results/multilingual_baseline/"
    )


# ================================================================
# STEP 2 - LOAD MINORITY-AWARE MODEL
# ================================================================

print_header(
    "2. LOADING STEP 20 - MINORITY-AWARE XLM-R"
)

minority_df = load_prediction_file(
    MINORITY_DIR,
    "Minority-Aware XLM-R"
)

if minority_df is None:
    raise FileNotFoundError(
        "\nStep 20 prediction CSV was not found.\n"
        "Check results/multilingual_minority_aware/"
    )


# ================================================================
# STEP 3 - PREPARE DATA
# ================================================================

baseline = prepare_predictions(
    baseline_df,
    "Multilingual XLM-R Baseline"
)

minority = prepare_predictions(
    minority_df,
    "Minority-Aware XLM-R"
)

baseline = convert_binary_labels(baseline)
minority = convert_binary_labels(minority)


# ================================================================
# STEP 4 - OVERALL PERFORMANCE
# ================================================================

print_header("3. OVERALL MODEL PERFORMANCE")

baseline_metrics = calculate_metrics(
    baseline["true_numeric"],
    baseline["prediction_numeric"]
)

minority_metrics = calculate_metrics(
    minority["true_numeric"],
    minority["prediction_numeric"]
)

overall_results = pd.DataFrame(
    [
        {
            "Model": "Multilingual XLM-R Baseline",
            **baseline_metrics
        },
        {
            "Model": "Minority-Aware XLM-R",
            **minority_metrics
        }
    ]
)

print(
    overall_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

overall_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "overall_model_comparison.csv"
    ),
    index=False
)


# ================================================================
# STEP 5 - PERFORMANCE IMPROVEMENT
# ================================================================

print_header("4. PERFORMANCE IMPROVEMENT")

improvement = []

for metric in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]:

    baseline_value = overall_results.loc[
        overall_results["Model"]
        == "Multilingual XLM-R Baseline",
        metric
    ].iloc[0]

    minority_value = overall_results.loc[
        overall_results["Model"]
        == "Minority-Aware XLM-R",
        metric
    ].iloc[0]

    absolute_change = minority_value - baseline_value

    improvement.append(
        {
            "Metric": metric,
            "Baseline": baseline_value,
            "Minority_Aware": minority_value,
            "Absolute_Change": absolute_change,
            "Percentage_Change": (
                (absolute_change / baseline_value) * 100
                if baseline_value != 0
                else 0
            )
        }
    )

improvement_df = pd.DataFrame(improvement)

print(
    improvement_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

improvement_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "performance_improvement.csv"
    ),
    index=False
)


# ================================================================
# STEP 6 - LANGUAGE-WISE PERFORMANCE
# ================================================================

print_header("5. LANGUAGE-WISE PERFORMANCE")

language_results = []

languages = sorted(
    set(baseline["language"].unique())
    | set(minority["language"].unique())
)

for language in languages:

    baseline_lang = baseline[
        baseline["language"].str.lower()
        == language.lower()
    ]

    minority_lang = minority[
        minority["language"].str.lower()
        == language.lower()
    ]

    if len(baseline_lang) > 0:

        metrics = calculate_metrics(
            baseline_lang["true_numeric"],
            baseline_lang["prediction_numeric"]
        )

        language_results.append(
            {
                "Language": language,
                "Model": "Multilingual XLM-R Baseline",
                "Records": len(baseline_lang),
                **metrics
            }
        )

    if len(minority_lang) > 0:

        metrics = calculate_metrics(
            minority_lang["true_numeric"],
            minority_lang["prediction_numeric"]
        )

        language_results.append(
            {
                "Language": language,
                "Model": "Minority-Aware XLM-R",
                "Records": len(minority_lang),
                **metrics
            }
        )


language_results_df = pd.DataFrame(language_results)

print(
    language_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

language_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "language_wise_model_comparison.csv"
    ),
    index=False
)


# ================================================================
# STEP 7 - OFFENSIVE CLASS PERFORMANCE
# ================================================================

print_header(
    "6. OFFENSIVE-CLASS PERFORMANCE"
)

offensive_results = []

for model_name, df in [
    ("Multilingual XLM-R Baseline", baseline),
    ("Minority-Aware XLM-R", minority)
]:

    y_true = df["true_numeric"]
    y_pred = df["prediction_numeric"]

    # Offensive class = 1
    offensive_precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    offensive_recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    offensive_f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    offensive_support = int(
        (y_true == 1).sum()
    )

    offensive_results.append(
        {
            "Model": model_name,
            "Offensive_Support": offensive_support,
            "Offensive_Precision": offensive_precision,
            "Offensive_Recall": offensive_recall,
            "Offensive_F1": offensive_f1
        }
    )

offensive_df = pd.DataFrame(
    offensive_results
)

print(
    offensive_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

offensive_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "offensive_class_comparison.csv"
    ),
    index=False
)


# ================================================================
# STEP 8 - CONFUSION MATRICES
# ================================================================

print_header("7. CONFUSION MATRICES")

models_data = [
    (
        "Multilingual XLM-R Baseline",
        baseline
    ),
    (
        "Minority-Aware XLM-R",
        minority
    )
]

for model_name, df in models_data:

    cm = confusion_matrix(
        df["true_numeric"],
        df["prediction_numeric"],
        labels=[0, 1]
    )

    print()
    print(model_name)
    print(cm)

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Non-Offensive",
            "Actual Offensive"
        ],
        columns=[
            "Predicted Non-Offensive",
            "Predicted Offensive"
        ]
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    cm_df.to_csv(
        os.path.join(
            REPORT_OUTPUT_DIR,
            f"{safe_name}_confusion_matrix.csv"
        )
    )


# ================================================================
# STEP 9 - CLASSIFICATION REPORTS
# ================================================================

print_header("8. CLASSIFICATION REPORTS")

for model_name, df in models_data:

    report = classification_report(
        df["true_numeric"],
        df["prediction_numeric"],
        target_names=[
            "Non_Offensive",
            "Offensive"
        ],
        zero_division=0
    )

    print()
    print(model_name)
    print("-" * 50)
    print(report)

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    with open(
        os.path.join(
            REPORT_OUTPUT_DIR,
            f"{safe_name}_classification_report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)


# ================================================================
# STEP 10 - MODEL PERFORMANCE PLOT
# ================================================================

print_header(
    "9. GENERATING MODEL COMPARISON VISUALIZATION"
)

metrics_to_plot = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]

x = np.arange(len(metrics_to_plot))
width = 0.35

baseline_values = [
    overall_results.loc[
        overall_results["Model"]
        == "Multilingual XLM-R Baseline",
        metric
    ].iloc[0]
    for metric in metrics_to_plot
]

minority_values = [
    overall_results.loc[
        overall_results["Model"]
        == "Minority-Aware XLM-R",
        metric
    ].iloc[0]
    for metric in metrics_to_plot
]

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Multilingual XLM-R Baseline"
)

plt.bar(
    x + width / 2,
    minority_values,
    width,
    label="Minority-Aware XLM-R"
)

plt.xticks(
    x,
    metrics_to_plot
)

plt.ylabel("Score")
plt.xlabel("Evaluation Metric")
plt.title(
    "Multilingual XLM-R Model Comparison"
)

plt.ylim(0, 1)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_OUTPUT_DIR,
        "overall_model_comparison.png"
    ),
    dpi=300
)

plt.close()

print(
    "Overall comparison plot generated."
)


# ================================================================
# STEP 11 - LANGUAGE-WISE F1 PLOT
# ================================================================

if not language_results_df.empty:

    pivot = language_results_df.pivot(
        index="Language",
        columns="Model",
        values="F1_Score"
    )

    plt.figure(figsize=(10, 6))

    pivot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.ylabel("F1 Score")
    plt.xlabel("Language")
    plt.title(
        "Language-wise F1 Score Comparison"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_OUTPUT_DIR,
            "language_wise_f1_comparison.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "Language-wise F1 plot generated."
    )


# ================================================================
# STEP 12 - SAVE COMBINED PREDICTIONS
# ================================================================

print_header(
    "10. SAVING COMBINED PREDICTION DATA"
)

combined_predictions = pd.concat(
    [
        baseline,
        minority
    ],
    ignore_index=True
)

combined_predictions.to_csv(
    os.path.join(
        PREDICTION_OUTPUT_DIR,
        "combined_model_predictions.csv"
    ),
    index=False
)

print(
    "Combined prediction file saved."
)


# ================================================================
# STEP 13 - FINAL SUMMARY
# ================================================================

print_header(
    "11. FINAL MODEL COMPARISON SUMMARY"
)

best_model = overall_results.loc[
    overall_results["F1_Score"].idxmax(),
    "Model"
]

best_f1 = overall_results.loc[
    overall_results["F1_Score"].idxmax(),
    "F1_Score"
]

print(
    overall_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("Best model based on F1-score:")
print(best_model)

print(
    f"Best F1-score: {best_f1:.4f}"
)


# ================================================================
# FINAL OUTPUT
# ================================================================

print_header(
    "STEP 21 - MULTILINGUAL MODEL COMPARISON COMPLETED"
)

print("All results saved in:")
print(OUTPUT_DIR)

print()
print("Generated files:")

print("- overall_model_comparison.csv")
print("- performance_improvement.csv")
print("- language_wise_model_comparison.csv")
print("- offensive_class_comparison.csv")
print("- predictions/combined_model_predictions.csv")
print("- reports/*_classification_report.txt")
print("- reports/*_confusion_matrix.csv")
print("- plots/overall_model_comparison.png")
print("- plots/language_wise_f1_comparison.png")

print()
print("=" * 70)
print("NO MODEL TRAINING WAS PERFORMED")
print("=" * 70)

print()
print(
    "Step 21 compares the multilingual XLM-R baseline "
    "with the minority-aware XLM-R model using the same "
    "evaluation framework."
)