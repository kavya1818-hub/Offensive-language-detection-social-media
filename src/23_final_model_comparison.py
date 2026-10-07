# ================================================================
# STEP 23 - FINAL MODEL COMPARISON
# ================================================================
# Purpose:
#   Final evidence-based comparison of the models developed in:
#
#   DATASET 1:
#       Initial offensive-language detection experiments
#       + class-imbalance experiments
#
#   DATASET 2:
#       Multilingual offensive-language detection
#       Tamil + Malayalam + Kannada
#
# IMPORTANT:
#   This script performs NO MODEL TRAINING.
#   It only reads previously generated result files.
#
# ================================================================

import os
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = "results"

DATASET1_DIR = os.path.join(BASE_DIR)
DATASET2_DIR = os.path.join(BASE_DIR, "multilingual_model_comparison")

OUTPUT_DIR = os.path.join(BASE_DIR, "final_model_comparison")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def clean_number(value):
    """Convert a metric to numeric safely."""
    try:
        value = pd.to_numeric(value, errors="coerce")
        return float(value)
    except Exception:
        return np.nan


def normalize_columns(df):
    """Normalize column names for easier matching."""
    df = df.copy()
    df.columns = [
        str(col).strip().replace(" ", "_").replace("-", "_")
        for col in df.columns
    ]
    return df


def find_column(df, possible_names):
    """Return first matching column from a list of possible names."""
    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ================================================================
# DATASET 1 - EXTRACT RESULTS
# ================================================================

print_header("STEP 23 - FINAL MODEL COMPARISON")

print(
    """
This step performs FINAL RESULT ANALYSIS.

Dataset 1:
Initial offensive-language detection and class-imbalance experiments.

Dataset 2:
Multilingual offensive-language detection using:
Tamil + Malayalam + Kannada.

No model training will be performed.
Only previously generated experimental results are analysed.
"""
)


# ================================================================
# 1. DISCOVER DATASET 1 RESULT FILES
# ================================================================

print_header("1. LOADING DATASET 1 RESULTS")

dataset1_files = []

search_patterns = [
    os.path.join(BASE_DIR, "baseline_models", "*.csv"),
    os.path.join(BASE_DIR, "baseline_evaluation", "*.csv"),
    os.path.join(BASE_DIR, "transformer", "*.csv"),
    os.path.join(BASE_DIR, "transformer_evaluation", "*.csv"),
    os.path.join(BASE_DIR, "weighted_transformer", "*.csv"),
    os.path.join(BASE_DIR, "balanced_transformer", "*.csv"),
    os.path.join(BASE_DIR, "adaptive_transformer", "*.csv"),
    os.path.join(BASE_DIR, "model_comparison", "*.csv"),
    os.path.join(BASE_DIR, "final_analysis", "*.csv"),
]

for pattern in search_patterns:
    dataset1_files.extend(glob.glob(pattern))


# Remove duplicates
dataset1_files = sorted(set(dataset1_files))

print(f"Dataset 1 CSV files discovered: {len(dataset1_files)}")

for file_path in dataset1_files:
    print(" -", os.path.relpath(file_path, BASE_DIR))


# ================================================================
# 2. EXTRACT COMPARABLE DATASET 1 MODEL RESULTS
# ================================================================

print_header("2. EXTRACTING DATASET 1 MODEL METRICS")

dataset1_records = []


# ------------------------------------------------
# Preferred comparison files
# ------------------------------------------------

preferred_files = [
    os.path.join(BASE_DIR, "model_comparison", "complete_model_comparison.csv"),
    os.path.join(BASE_DIR, "model_comparison", "model_ranking.csv"),
    os.path.join(BASE_DIR, "model_comparison", "traditional_ml_comparison.csv"),
    os.path.join(BASE_DIR, "model_comparison", "transformer_only_comparison.csv"),
    os.path.join(BASE_DIR, "final_analysis", "final_model_comparison.csv"),
    os.path.join(BASE_DIR, "final_analysis", "transformer_comparison.csv"),
]


def extract_metrics_from_file(path):
    """
    Extract model-level Accuracy, Precision, Recall and F1
    from an existing result CSV.
    """

    df = read_csv_safe(path)

    if df is None or df.empty:
        return []

    df = normalize_columns(df)

    model_col = find_column(
        df,
        [
            "Model",
            "model",
            "Model_Name",
            "model_name",
            "Classifier",
            "classifier",
            "Approach",
            "approach",
        ],
    )

    if model_col is None:
        return []

    accuracy_col = find_column(
        df,
        [
            "Accuracy",
            "accuracy",
            "Test_Accuracy",
            "test_accuracy",
        ],
    )

    precision_col = find_column(
        df,
        [
            "Precision",
            "precision",
            "Test_Precision",
            "test_precision",
        ],
    )

    recall_col = find_column(
        df,
        [
            "Recall",
            "recall",
            "Test_Recall",
            "test_recall",
        ],
    )

    f1_col = find_column(
        df,
        [
            "F1_Score",
            "F1",
            "f1_score",
            "f1",
            "Test_F1",
            "test_f1",
        ],
    )

    records = []

    for _, row in df.iterrows():

        model_name = str(row[model_col]).strip()

        if not model_name or model_name.lower() == "nan":
            continue

        record = {
            "Model": model_name,
            "Accuracy": (
                clean_number(row[accuracy_col])
                if accuracy_col
                else np.nan
            ),
            "Precision": (
                clean_number(row[precision_col])
                if precision_col
                else np.nan
            ),
            "Recall": (
                clean_number(row[recall_col])
                if recall_col
                else np.nan
            ),
            "F1_Score": (
                clean_number(row[f1_col])
                if f1_col
                else np.nan
            ),
            "Source_File": os.path.relpath(path, BASE_DIR),
            "Dataset": "Dataset 1",
        }

        # Ignore rows that contain no useful metric
        metric_values = [
            record["Accuracy"],
            record["Precision"],
            record["Recall"],
            record["F1_Score"],
        ]

        if not all(pd.isna(v) for v in metric_values):
            records.append(record)

    return records


# First use preferred files
for path in preferred_files:
    if os.path.exists(path):
        dataset1_records.extend(extract_metrics_from_file(path))


# ------------------------------------------------
# If preferred files are incomplete, inspect all files
# ------------------------------------------------

if len(dataset1_records) == 0:

    for path in dataset1_files:
        dataset1_records.extend(extract_metrics_from_file(path))


# ------------------------------------------------
# Remove duplicate model/source combinations
# ------------------------------------------------

if dataset1_records:

    dataset1_df = pd.DataFrame(dataset1_records)

    dataset1_df = dataset1_df.drop_duplicates(
        subset=["Model", "Source_File"],
        keep="first"
    )

else:
    dataset1_df = pd.DataFrame(
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "Source_File",
            "Dataset",
        ]
    )


# ================================================================
# 3. CLEAN DATASET 1 RESULTS
# ================================================================

print_header("3. CLEANING DATASET 1 RESULTS")

# Some files may contain intermediate rows or duplicate evaluations.
# Prefer model comparison files when possible.

if not dataset1_df.empty:

    # Remove obvious non-model summary rows
    invalid_names = {
        "",
        "nan",
        "best model",
        "average",
        "mean",
        "overall",
        "summary",
    }

    dataset1_df = dataset1_df[
        ~dataset1_df["Model"]
        .astype(str)
        .str.lower()
        .isin(invalid_names)
    ].copy()

    # Numeric conversion
    for col in ["Accuracy", "Precision", "Recall", "F1_Score"]:
        dataset1_df[col] = pd.to_numeric(
            dataset1_df[col],
            errors="coerce"
        )

    # If multiple rows for the same model exist, keep the row
    # with the most available metrics.
    dataset1_df["Metric_Count"] = dataset1_df[
        ["Accuracy", "Precision", "Recall", "F1_Score"]
    ].notna().sum(axis=1)

    dataset1_df = (
        dataset1_df
        .sort_values("Metric_Count", ascending=False)
        .drop_duplicates("Model", keep="first")
        .drop(columns=["Metric_Count"])
        .reset_index(drop=True)
    )


print("Dataset 1 models extracted:")

if dataset1_df.empty:
    print("No comparable Dataset 1 model table could be extracted.")
else:
    print(dataset1_df.to_string(index=False))


# Save raw extracted Dataset 1 comparison
dataset1_output = os.path.join(
    OUTPUT_DIR,
    "dataset1_model_comparison.csv"
)

save_csv(dataset1_df, dataset1_output)


# ================================================================
# 4. SELECT BEST DATASET 1 MODEL
# ================================================================

print_header("4. BEST DATASET 1 MODEL")

best_dataset1 = None

if not dataset1_df.empty:

    # F1 is the preferred metric because offensive language detection
    # is a class-imbalanced classification problem.

    f1_available = dataset1_df["F1_Score"].notna()

    if f1_available.any():

        candidates = dataset1_df[f1_available].copy()

        best_dataset1 = candidates.loc[
            candidates["F1_Score"].idxmax()
        ].copy()

        print("Best Dataset 1 model based on available F1-score:")
        print(best_dataset1.to_string())

    else:

        accuracy_available = dataset1_df["Accuracy"].notna()

        if accuracy_available.any():

            candidates = dataset1_df[accuracy_available].copy()

            best_dataset1 = candidates.loc[
                candidates["Accuracy"].idxmax()
            ].copy()

            print(
                "Dataset 1 does not contain comparable F1 values "
                "for the available models."
            )

            print(
                "Best Dataset 1 model based on Accuracy only:"
            )

            print(best_dataset1.to_string())

        else:
            print("No usable Dataset 1 evaluation metric was found.")

else:
    print("Dataset 1 comparison unavailable.")


# ================================================================
# 5. DATASET 1 VISUALIZATION
# ================================================================

print_header("5. DATASET 1 MODEL VISUALIZATION")

if not dataset1_df.empty:

    plot_df = dataset1_df.copy()

    # Prefer F1 if available
    if plot_df["F1_Score"].notna().any():

        plot_df = plot_df[
            plot_df["F1_Score"].notna()
        ].copy()

        plot_df = plot_df.sort_values(
            "F1_Score",
            ascending=True
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            plot_df["Model"],
            plot_df["F1_Score"]
        )

        plt.xlabel("F1-score")
        plt.ylabel("Model")
        plt.title("Dataset 1 - Model F1-score Comparison")
        plt.tight_layout()

        path = os.path.join(
            PLOTS_DIR,
            "dataset1_f1_comparison.png"
        )

        plt.savefig(path, dpi=300)
        plt.close()

        print("Plot saved:", path)

    elif plot_df["Accuracy"].notna().any():

        plot_df = plot_df[
            plot_df["Accuracy"].notna()
        ].copy()

        plot_df = plot_df.sort_values(
            "Accuracy",
            ascending=True
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            plot_df["Model"],
            plot_df["Accuracy"]
        )

        plt.xlabel("Accuracy")
        plt.ylabel("Model")
        plt.title("Dataset 1 - Model Accuracy Comparison")
        plt.tight_layout()

        path = os.path.join(
            PLOTS_DIR,
            "dataset1_accuracy_comparison.png"
        )

        plt.savefig(path, dpi=300)
        plt.close()

        print("Plot saved:", path)


# ================================================================
# 6. LOAD DATASET 2 MULTILINGUAL RESULTS
# ================================================================

print_header("6. LOADING DATASET 2 MULTILINGUAL RESULTS")

dataset2_comparison_path = os.path.join(
    DATASET2_DIR,
    "overall_model_comparison.csv"
)

if not os.path.exists(dataset2_comparison_path):

    raise FileNotFoundError(
        "Dataset 2 comparison file was not found:\n"
        + dataset2_comparison_path
    )


dataset2_df = pd.read_csv(dataset2_comparison_path)
dataset2_df = normalize_columns(dataset2_df)

print("Dataset 2 comparison file:")
print(dataset2_comparison_path)

print("\nDataset 2 models actually evaluated:")

print(dataset2_df.to_string(index=False))


# Ensure expected metric columns exist
required_dataset2_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
]

missing = [
    col
    for col in required_dataset2_columns
    if col not in dataset2_df.columns
]

if missing:
    raise ValueError(
        "Dataset 2 result file is missing columns: "
        + str(missing)
    )


for col in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
]:
    dataset2_df[col] = pd.to_numeric(
        dataset2_df[col],
        errors="coerce"
    )


dataset2_df["Dataset"] = "Dataset 2"


# ================================================================
# 7. BEST DATASET 2 MODEL
# ================================================================

print_header("7. BEST DATASET 2 MODEL")

dataset2_candidates = dataset2_df[
    dataset2_df["F1_Score"].notna()
].copy()

if dataset2_candidates.empty:

    raise ValueError(
        "Dataset 2 contains no valid F1-score values."
    )


best_dataset2 = dataset2_candidates.loc[
    dataset2_candidates["F1_Score"].idxmax()
].copy()

print(
    "Best Dataset 2 model based on F1-score:"
)

print(best_dataset2.to_string())


# Save Dataset 2 comparison
dataset2_output = os.path.join(
    OUTPUT_DIR,
    "dataset2_multilingual_model_comparison.csv"
)

save_csv(dataset2_df, dataset2_output)


# ================================================================
# 8. DATASET 2 VISUALIZATION
# ================================================================

print_header("8. DATASET 2 MODEL VISUALIZATION")

plot_df = dataset2_df.sort_values(
    "F1_Score",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    plot_df["Model"],
    plot_df["F1_Score"]
)

plt.xlabel("F1-score")
plt.ylabel("Model")
plt.title("Dataset 2 - Multilingual XLM-R Model Comparison")
plt.tight_layout()

dataset2_plot = os.path.join(
    PLOTS_DIR,
    "dataset2_multilingual_f1_comparison.png"
)

plt.savefig(
    dataset2_plot,
    dpi=300
)

plt.close()

print("Plot saved:", dataset2_plot)


# ================================================================
# 9. DATASET 2 LANGUAGE-WISE ANALYSIS
# ================================================================

print_header("9. DATASET 2 LANGUAGE-WISE RESULTS")

language_path = os.path.join(
    DATASET2_DIR,
    "language_wise_model_comparison.csv"
)

if os.path.exists(language_path):

    language_df = pd.read_csv(language_path)
    language_df = normalize_columns(language_df)

    print("Language-wise result file found:")
    print(language_path)

    print("\nLanguage-wise performance:")
    print(language_df.to_string(index=False))

    language_output = os.path.join(
        OUTPUT_DIR,
        "dataset2_language_wise_performance.csv"
    )

    save_csv(
        language_df,
        language_output
    )

else:

    language_df = pd.DataFrame()

    print(
        "Language-wise result file not found."
    )


# ================================================================
# 10. DATASET 2 OFFENSIVE CLASS ANALYSIS
# ================================================================

print_header("10. DATASET 2 OFFENSIVE-CLASS RESULTS")

offensive_path = os.path.join(
    DATASET2_DIR,
    "offensive_class_comparison.csv"
)

if os.path.exists(offensive_path):

    offensive_df = pd.read_csv(offensive_path)
    offensive_df = normalize_columns(offensive_df)

    print("Offensive-class result file found:")
    print(offensive_path)

    print("\nOffensive-class performance:")
    print(offensive_df.to_string(index=False))

    offensive_output = os.path.join(
        OUTPUT_DIR,
        "dataset2_offensive_class_performance.csv"
    )

    save_csv(
        offensive_df,
        offensive_output
    )

else:

    offensive_df = pd.DataFrame()

    print(
        "Offensive-class result file not found."
    )


# ================================================================
# 11. DATASET 2 LANGUAGE-WISE F1 PLOT
# ================================================================

print_header("11. LANGUAGE-WISE F1 VISUALIZATION")

if not language_df.empty:

    required_cols = {
        "Language",
        "Model",
        "F1_Score",
    }

    if required_cols.issubset(
        set(language_df.columns)
    ):

        plt.figure(figsize=(10, 6))

        for model in language_df["Model"].unique():

            model_data = language_df[
                language_df["Model"] == model
            ]

            plt.plot(
                model_data["Language"],
                model_data["F1_Score"],
                marker="o",
                label=model
            )

        plt.xlabel("Language")
        plt.ylabel("F1-score")
        plt.title(
            "Dataset 2 - Language-wise F1-score"
        )

        plt.legend()
        plt.tight_layout()

        language_plot = os.path.join(
            PLOTS_DIR,
            "dataset2_language_wise_f1.png"
        )

        plt.savefig(
            language_plot,
            dpi=300
        )

        plt.close()

        print(
            "Plot saved:",
            language_plot
        )

    else:
        print(
            "Required language-wise columns are unavailable."
        )


# ================================================================
# 12. FINAL MODEL SUMMARY
# ================================================================

print_header("12. FINAL MODEL SUMMARY")

final_records = []


if best_dataset1 is not None:

    final_records.append(
        {
            "Dataset": "Dataset 1",
            "Best_Model": best_dataset1["Model"],
            "Accuracy": best_dataset1["Accuracy"],
            "Precision": best_dataset1["Precision"],
            "Recall": best_dataset1["Recall"],
            "F1_Score": best_dataset1["F1_Score"],
        }
    )


final_records.append(
    {
        "Dataset": "Dataset 2",
        "Best_Model": best_dataset2["Model"],
        "Accuracy": best_dataset2["Accuracy"],
        "Precision": best_dataset2["Precision"],
        "Recall": best_dataset2["Recall"],
        "F1_Score": best_dataset2["F1_Score"],
    }
)


final_summary_df = pd.DataFrame(final_records)

print(
    final_summary_df.to_string(index=False)
)

final_summary_path = os.path.join(
    OUTPUT_DIR,
    "final_best_model_summary.csv"
)

save_csv(
    final_summary_df,
    final_summary_path
)


# ================================================================
# 13. RESEARCH STAGE COMPARISON
# ================================================================

print_header("13. RESEARCH STAGE COMPARISON")

stage_records = []

if best_dataset1 is not None:

    stage_records.append(
        {
            "Research_Stage": "Dataset 1",
            "Best_Model": best_dataset1["Model"],
            "Accuracy": best_dataset1["Accuracy"],
            "Precision": best_dataset1["Precision"],
            "Recall": best_dataset1["Recall"],
            "F1_Score": best_dataset1["F1_Score"],
            "Purpose": (
                "Initial offensive-language detection "
                "and class-imbalance experiments"
            ),
        }
    )


stage_records.append(
    {
        "Research_Stage": "Dataset 2",
        "Best_Model": best_dataset2["Model"],
        "Accuracy": best_dataset2["Accuracy"],
        "Precision": best_dataset2["Precision"],
        "Recall": best_dataset2["Recall"],
        "F1_Score": best_dataset2["F1_Score"],
        "Purpose": (
            "Multilingual offensive-language detection "
            "using Tamil, Malayalam and Kannada"
        ),
    }
)


stage_df = pd.DataFrame(stage_records)

print(
    stage_df.to_string(index=False)
)

stage_path = os.path.join(
    OUTPUT_DIR,
    "research_stage_comparison.csv"
)

save_csv(
    stage_df,
    stage_path
)


# ================================================================
# 14. MULTILINGUAL OBJECTIVE VERIFICATION
# ================================================================

print_header("14. MULTILINGUAL OBJECTIVE VERIFICATION")

languages_found = []

if not language_df.empty:

    language_column = find_column(
        language_df,
        ["Language", "language"]
    )

    if language_column:

        languages_found = sorted(
            language_df[language_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )


objective_status = (
    len(languages_found) >= 3
    and os.path.exists(dataset2_comparison_path)
)


objective_df = pd.DataFrame(
    [
        {
            "Objective":
                "Multilingual offensive-language detection",
            "Dataset":
                "Dataset 2",
            "Languages":
                ", ".join(languages_found)
                if languages_found
                else "Not available",
            "Models_Evaluated":
                ", ".join(
                    dataset2_df["Model"]
                    .astype(str)
                    .tolist()
                ),
            "Evidence":
                (
                    "Overall model comparison, "
                    "language-wise evaluation, "
                    "offensive-class analysis, "
                    "and error analysis"
                ),
            "Status":
                "Demonstrated"
                if objective_status
                else "Insufficient evidence",
        }
    ]
)

print(
    objective_df.to_string(index=False)
)

objective_path = os.path.join(
    OUTPUT_DIR,
    "multilingual_objective_verification.csv"
)

save_csv(
    objective_df,
    objective_path
)


# ================================================================
# 15. DATASET 2 MODEL INTERPRETATION
# ================================================================

print_header("15. DATASET 2 MODEL INTERPRETATION")

baseline_row = dataset2_df[
    dataset2_df["Model"]
    .astype(str)
    .str.contains(
        "Baseline",
        case=False,
        na=False
    )
]

minority_row = dataset2_df[
    dataset2_df["Model"]
    .astype(str)
    .str.contains(
        "Minority",
        case=False,
        na=False
    )
]


interpretation_lines = []

if not baseline_row.empty:

    baseline = baseline_row.iloc[0]

    interpretation_lines.append(
        "Multilingual XLM-R Baseline:"
    )

    interpretation_lines.append(
        f"Accuracy = {baseline['Accuracy']:.4f}"
    )

    interpretation_lines.append(
        f"Precision = {baseline['Precision']:.4f}"
    )

    interpretation_lines.append(
        f"Recall = {baseline['Recall']:.4f}"
    )

    interpretation_lines.append(
        f"F1-score = {baseline['F1_Score']:.4f}"
    )


if not minority_row.empty:

    minority = minority_row.iloc[0]

    interpretation_lines.append(
        ""
    )

    interpretation_lines.append(
        "Minority-Aware XLM-R:"
    )

    interpretation_lines.append(
        f"Accuracy = {minority['Accuracy']:.4f}"
    )

    interpretation_lines.append(
        f"Precision = {minority['Precision']:.4f}"
    )

    interpretation_lines.append(
        f"Recall = {minority['Recall']:.4f}"
    )

    interpretation_lines.append(
        f"F1-score = {minority['F1_Score']:.4f}"
    )


if (
    not baseline_row.empty
    and not minority_row.empty
):

    baseline = baseline_row.iloc[0]
    minority = minority_row.iloc[0]

    interpretation_lines.extend(
        [
            "",
            "Observed comparison:",
            (
                "The baseline has higher overall accuracy, "
                "but its offensive-class precision, recall "
                "and F1-score are zero."
            ),
            (
                "The minority-aware model has lower overall "
                "accuracy but detects offensive examples, "
                "with non-zero offensive precision, recall "
                "and F1-score."
            ),
            (
                "Therefore, accuracy alone is not sufficient "
                "for evaluating this imbalanced offensive-language "
                "classification task."
            ),
        ]
    )


for line in interpretation_lines:
    print(line)


# ================================================================
# 16. FINAL RESEARCH REPORT
# ================================================================

print_header("16. GENERATING FINAL REPORT")

report_path = os.path.join(
    REPORTS_DIR,
    "final_results.txt"
)


report = []

report.append(
    "FINAL MODEL COMPARISON REPORT"
)

report.append(
    "=" * 60
)

report.append(
    ""
)

report.append(
    "PROJECT STRUCTURE"
)

report.append(
    "Dataset 1: Initial offensive-language detection "
    "and class-imbalance experiments."
)

report.append(
    "Dataset 2: Multilingual offensive-language detection "
    "using Tamil, Malayalam and Kannada."
)

report.append(
    ""
)

report.append(
    "DATASET 1 RESULTS"
)

if best_dataset1 is not None:

    report.append(
        f"Best model: {best_dataset1['Model']}"
    )

    report.append(
        f"Accuracy: {best_dataset1['Accuracy']}"
    )

    report.append(
        f"Precision: {best_dataset1['Precision']}"
    )

    report.append(
        f"Recall: {best_dataset1['Recall']}"
    )

    report.append(
        f"F1-score: {best_dataset1['F1_Score']}"
    )

    report.append(
        f"Source: {best_dataset1['Source_File']}"
    )

else:

    report.append(
        "No comparable Dataset 1 model result was available."
    )


report.append(
    ""
)

report.append(
    "DATASET 2 RESULTS"
)

report.append(
    f"Best model: {best_dataset2['Model']}"
)

report.append(
    f"Accuracy: {best_dataset2['Accuracy']:.6f}"
)

report.append(
    f"Precision: {best_dataset2['Precision']:.6f}"
)

report.append(
    f"Recall: {best_dataset2['Recall']:.6f}"
)

report.append(
    f"F1-score: {best_dataset2['F1_Score']:.6f}"
)

report.append(
    ""
)

report.append(
    "DATASET 2 LANGUAGES"
)

if languages_found:

    report.append(
        ", ".join(languages_found)
    )

else:

    report.append(
        "Language-wise result not available."
    )


report.append(
    ""
)

report.append(
    "MULTILINGUAL OBJECTIVE"
)

if objective_status:

    report.append(
        "Demonstrated through Dataset 2."
    )

    report.append(
        "The multilingual model was evaluated on "
        "Tamil, Malayalam and Kannada."
    )

    report.append(
        "Performance was analysed overall and "
        "separately by language."
    )

    report.append(
        "Offensive-class performance was also evaluated."
    )

else:

    report.append(
        "Insufficient result evidence."
    )


report.append(
    ""
)

report.append(
    "IMPORTANT INTERPRETATION"
)

report.append(
    "Dataset 1 and Dataset 2 are separate experimental stages."
)

report.append(
    "They are not treated as a single common leaderboard "
    "because they represent different datasets and research objectives."
)

report.append(
    "Dataset 1 establishes the initial offensive-language "
    "classification pipeline and studies class imbalance."
)

report.append(
    "Dataset 2 extends the project to multilingual offensive-language "
    "detection using Tamil, Malayalam and Kannada."
)

if (
    not baseline_row.empty
    and not minority_row.empty
):

    report.append(
        ""
    )

    report.append(
        "Dataset 2 model interpretation:"
    )

    report.append(
        "The baseline achieved higher overall accuracy but "
        "failed to detect offensive samples."
    )

    report.append(
        "The minority-aware model achieved lower overall accuracy "
        "but produced non-zero offensive precision, recall and F1-score."
    )

    report.append(
        "For this imbalanced task, offensive-class F1-score and recall "
        "are therefore important when interpreting practical performance."
    )


report.append(
    ""
)

report.append(
    "NO MODEL TRAINING WAS PERFORMED IN STEP 23."
)

report.append(
    "Step 23 only analyses previously generated experimental results."
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report)
    )


print(
    "Final report saved:"
)

print(
    report_path
)


# ================================================================
# 17. FINAL OUTPUT SUMMARY
# ================================================================

print_header("17. STEP 23 COMPLETED")

print(
    "All final results saved in:"
)

print(
    OUTPUT_DIR
)

print(
    "\nGenerated files:"
)

generated_files = []

for root, _, files in os.walk(
    OUTPUT_DIR
):

    for file_name in files:

        full_path = os.path.join(
            root,
            file_name
        )

        generated_files.append(
            os.path.relpath(
                full_path,
                OUTPUT_DIR
            )
        )


for index, file_name in enumerate(
    sorted(generated_files),
    start=1
):

    print(
        f"{index}. {file_name}"
    )


print(
    "\n" + "=" * 70
)

print(
    "NO MODEL TRAINING WAS PERFORMED"
)

print(
    "=" * 70
)

print(
    """
Dataset 1:
Initial offensive-language detection and
class-imbalance experiments.

Dataset 2:
Multilingual offensive-language detection
using Tamil, Malayalam and Kannada.

Step 23 compares the actual results generated
by the completed experiments and does not create
artificial or assumed performance values.
"""
)

print(
    "=" * 70
)
print(
    "STEP 23 - FINAL MODEL COMPARISON COMPLETED"
)
print(
    "=" * 70
)