# ============================================================
# STEP 12 - COMPLETE MODEL COMPARISON
# Traditional ML vs XLM-R Transformer Models
# ============================================================

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = "results"

BASELINE_DIR = os.path.join(
    RESULTS_DIR,
    "baseline_evaluation"
)

TRANSFORMER_DIR = os.path.join(
    RESULTS_DIR,
    "transformer"
)

WEIGHTED_DIR = os.path.join(
    RESULTS_DIR,
    "weighted_transformer"
)

BALANCED_DIR = os.path.join(
    RESULTS_DIR,
    "balanced_transformer"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "model_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("STEP 12 - COMPLETE MODEL COMPARISON")
print("=" * 70)

print("\nIMPORTANT:")
print("Traditional ML results are read from baseline_evaluation")
print("Classification reports are used as the authoritative test results.")
print("results/model_evaluation.csv is NOT used.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_columns(df):
    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df


def find_existing_file(paths):

    for path in paths:

        if os.path.exists(path):
            return path

    return None


def find_baseline_report(model_keywords):

    """
    Search only inside:
    results/baseline_evaluation/

    This prevents accidentally loading the wrong
    model_evaluation.csv file.
    """

    all_csv_files = glob.glob(
        os.path.join(
            BASELINE_DIR,
            "*.csv"
        )
    )

    for file_path in all_csv_files:

        filename = os.path.basename(
            file_path
        ).lower()

        if "classification_report" not in filename:
            continue

        if all(
            keyword.lower() in filename
            for keyword in model_keywords
        ):
            return file_path

    return None


def read_classification_report(
    file_path,
    model_name
):

    """
    Read sklearn classification_report CSV.

    Expected rows:
    Hate_Speech
    Neither
    Offensive_Language
    accuracy
    macro avg
    weighted avg
    """

    print("\nReading:")
    print(file_path)

    df = pd.read_csv(
        file_path,
        index_col=0
    )

    # Clean row names
    df.index = (
        df.index
        .astype(str)
        .str.strip()
    )

    print("Rows found:")
    print(list(df.index))

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    if "accuracy" in df.index:

        accuracy = float(
            df.loc[
                "accuracy",
                "f1-score"
            ]
        )

    else:

        raise ValueError(
            f"Accuracy row not found in {file_path}"
        )

    # --------------------------------------------------------
    # Macro metrics
    # --------------------------------------------------------

    if "macro avg" in df.index:

        macro_precision = float(
            df.loc[
                "macro avg",
                "precision"
            ]
        )

        macro_recall = float(
            df.loc[
                "macro avg",
                "recall"
            ]
        )

        macro_f1 = float(
            df.loc[
                "macro avg",
                "f1-score"
            ]
        )

    else:

        raise ValueError(
            f"macro avg row not found in {file_path}"
        )

    # --------------------------------------------------------
    # Weighted metrics
    # --------------------------------------------------------

    if "weighted avg" in df.index:

        weighted_precision = float(
            df.loc[
                "weighted avg",
                "precision"
            ]
        )

        weighted_recall = float(
            df.loc[
                "weighted avg",
                "recall"
            ]
        )

        weighted_f1 = float(
            df.loc[
                "weighted avg",
                "f1-score"
            ]
        )

    else:

        raise ValueError(
            f"weighted avg row not found in {file_path}"
        )

    return {

        "Category":
            "Traditional ML",

        "Model":
            model_name,

        "Accuracy":
            accuracy,

        "Macro Precision":
            macro_precision,

        "Macro Recall":
            macro_recall,

        "Macro F1":
            macro_f1,

        "Weighted Precision":
            weighted_precision,

        "Weighted Recall":
            weighted_recall,

        "Weighted F1":
            weighted_f1
    }


def read_transformer_result(
    file_path,
    model_name
):

    """
    Read the final TEST result generated by
    transformer training scripts.
    """

    print("\nReading:")
    print(file_path)

    df = pd.read_csv(
        file_path
    )

    print("Columns found:")
    print(list(df.columns))

    df = normalize_columns(
        df
    )

    # Usually one row
    row = df.iloc[0]

    def get_metric(
        possible_names
    ):

        for name in possible_names:

            name = (
                name
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            if name in df.columns:

                value = row[name]

                try:
                    return float(value)

                except:
                    return np.nan

        return np.nan

    return {

        "Category":
            "Transformer",

        "Model":
            model_name,

        "Accuracy":
            get_metric([
                "accuracy",
                "eval_accuracy"
            ]),

        "Macro Precision":
            get_metric([
                "macro_precision",
                "eval_macro_precision"
            ]),

        "Macro Recall":
            get_metric([
                "macro_recall",
                "eval_macro_recall"
            ]),

        "Macro F1":
            get_metric([
                "macro_f1",
                "eval_macro_f1"
            ]),

        "Weighted Precision":
            get_metric([
                "weighted_precision",
                "eval_weighted_precision"
            ]),

        "Weighted Recall":
            get_metric([
                "weighted_recall",
                "eval_weighted_recall"
            ]),

        "Weighted F1":
            get_metric([
                "weighted_f1",
                "eval_weighted_f1"
            ])
    }


# ============================================================
# STORAGE
# ============================================================

results = []


# ============================================================
# 1. TRADITIONAL ML MODELS
# ============================================================

print("\n" + "=" * 70)
print("1. LOADING TRADITIONAL ML TEST RESULTS")
print("=" * 70)

print("\nBaseline directory:")
print(BASELINE_DIR)


# ------------------------------------------------------------
# LINEAR SVM
# ------------------------------------------------------------

print("\n------------------------------------------------------------")
print("LINEAR SVM")
print("------------------------------------------------------------")

linear_svm_path = find_baseline_report(
    ["linear_svm"]
)

if linear_svm_path:

    result = read_classification_report(
        linear_svm_path,
        "Linear SVM"
    )

    results.append(
        result
    )

else:

    print(
        "WARNING: Linear SVM classification report not found."
    )


# ------------------------------------------------------------
# LOGISTIC REGRESSION
# ------------------------------------------------------------

print("\n------------------------------------------------------------")
print("LOGISTIC REGRESSION")
print("------------------------------------------------------------")

logistic_path = find_baseline_report(
    ["logistic_regression"]
)

if logistic_path:

    result = read_classification_report(
        logistic_path,
        "Logistic Regression"
    )

    results.append(
        result
    )

else:

    print(
        "WARNING: Logistic Regression classification report not found."
    )


# ------------------------------------------------------------
# MULTINOMIAL NAIVE BAYES
# ------------------------------------------------------------

print("\n------------------------------------------------------------")
print("MULTINOMIAL NAIVE BAYES")
print("------------------------------------------------------------")

naive_bayes_path = find_baseline_report(
    ["multinomial_naive_bayes"]
)

if naive_bayes_path:

    result = read_classification_report(
        naive_bayes_path,
        "Multinomial Naive Bayes"
    )

    results.append(
        result
    )

else:

    print(
        "WARNING: Multinomial Naive Bayes classification report not found."
    )


# ============================================================
# 2. XLM-R BASELINE
# ============================================================

print("\n" + "=" * 70)
print("2. LOADING BASELINE XLM-R")
print("=" * 70)

baseline_xlm_path = find_existing_file([

    os.path.join(
        TRANSFORMER_DIR,
        "xlm_roberta_results.csv"
    ),

    os.path.join(
        TRANSFORMER_DIR,
        "xlm_roberta_results.csv"
    )
])


if baseline_xlm_path:

    result = read_transformer_result(
        baseline_xlm_path,
        "XLM-R Baseline"
    )

    results.append(
        result
    )

else:

    print(
        "WARNING: Baseline XLM-R results not found."
    )


# ============================================================
# 3. CLASS-WEIGHTED XLM-R
# ============================================================

print("\n" + "=" * 70)
print("3. LOADING CLASS-WEIGHTED XLM-R")
print("=" * 70)

weighted_xlm_path = find_existing_file([

    os.path.join(
        WEIGHTED_DIR,
        "weighted_xlm_roberta_results.csv"
    )
])


if weighted_xlm_path:

    result = read_transformer_result(
        weighted_xlm_path,
        "XLM-R Weighted"
    )

    results.append(
        result
    )

else:

    print(
        "WARNING: Weighted XLM-R results not found."
    )


# ============================================================
# 4. BALANCED XLM-R
# ============================================================

print("\n" + "=" * 70)
print("4. LOADING BALANCED XLM-R")
print("=" * 70)

balanced_xlm_path = find_existing_file([

    os.path.join(
        BALANCED_DIR,
        "balanced_xlm_roberta_results.csv"
    )
])


if balanced_xlm_path:

    result = read_transformer_result(
        balanced_xlm_path,
        "XLM-R Balanced"
    )

    results.append(
        result
    )

else:

    print(
        "WARNING: Balanced XLM-R results not found."
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

print("\n" + "=" * 70)
print("CREATING FINAL MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame(
    results
)


if comparison.empty:

    print(
        "\nERROR: No model results were found."
    )

    print(
        "\nCheck these folders:"
    )

    print(
        "results/baseline_evaluation/"
    )

    print(
        "results/transformer/"
    )

    print(
        "results/weighted_transformer/"
    )

    print(
        "results/balanced_transformer/"
    )

    raise SystemExit


# ============================================================
# CLEAN NUMBERS
# ============================================================

metric_columns = [

    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted Precision",
    "Weighted Recall",
    "Weighted F1"
]


for column in metric_columns:

    comparison[column] = pd.to_numeric(
        comparison[column],
        errors="coerce"
    )

    comparison[column] = (
        comparison[column]
        .round(4)
    )


# ============================================================
# DISPLAY COMPLETE COMPARISON
# ============================================================

display_columns = [

    "Category",
    "Model",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1"
]


comparison_sorted = comparison.sort_values(
    by="Macro F1",
    ascending=False
)


print("\n")
print("=" * 110)
print("FINAL MODEL COMPARISON")
print("=" * 110)

print(
    comparison_sorted[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE COMPLETE COMPARISON
# ============================================================

complete_path = os.path.join(
    OUTPUT_DIR,
    "complete_model_comparison.csv"
)

comparison_sorted.to_csv(
    complete_path,
    index=False
)

print("\nComplete comparison saved:")
print(complete_path)


# ============================================================
# TRANSFORMER-ONLY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("TRANSFORMER-ONLY COMPARISON")
print("=" * 70)

transformer_comparison = comparison[
    comparison["Category"] == "Transformer"
].copy()


transformer_comparison = transformer_comparison.sort_values(
    by="Macro F1",
    ascending=False
)


print(
    transformer_comparison[
        display_columns
    ].to_string(
        index=False
    )
)


transformer_only_path = os.path.join(
    OUTPUT_DIR,
    "transformer_only_comparison.csv"
)

transformer_comparison.to_csv(
    transformer_only_path,
    index=False
)

print("\nTransformer-only comparison saved:")
print(transformer_only_path)


# ============================================================
# TRADITIONAL ML ONLY
# ============================================================

print("\n" + "=" * 70)
print("TRADITIONAL ML-ONLY COMPARISON")
print("=" * 70)

traditional_comparison = comparison[
    comparison["Category"] == "Traditional ML"
].copy()


traditional_comparison = traditional_comparison.sort_values(
    by="Macro F1",
    ascending=False
)


print(
    traditional_comparison[
        display_columns
    ].to_string(
        index=False
    )
)


traditional_path = os.path.join(
    OUTPUT_DIR,
    "traditional_ml_comparison.csv"
)

traditional_comparison.to_csv(
    traditional_path,
    index=False
)


# ============================================================
# RANKING
# ============================================================

comparison["Accuracy Rank"] = (
    comparison["Accuracy"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


comparison["Macro F1 Rank"] = (
    comparison["Macro F1"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


comparison["Weighted F1 Rank"] = (
    comparison["Weighted F1"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


ranking = comparison.sort_values(
    by=[
        "Macro F1",
        "Weighted F1"
    ],
    ascending=False
)


ranking_path = os.path.join(
    OUTPUT_DIR,
    "model_ranking.csv"
)

ranking.to_csv(
    ranking_path,
    index=False
)


# ============================================================
# BEST MODELS
# ============================================================

best_accuracy = comparison.loc[
    comparison["Accuracy"].idxmax()
]

best_macro_f1 = comparison.loc[
    comparison["Macro F1"].idxmax()
]

best_weighted_f1 = comparison.loc[
    comparison["Weighted F1"].idxmax()
]


print("\n" + "=" * 70)
print("BEST MODELS")
print("=" * 70)


print(
    f"\nBest Accuracy:"
    f"\n{best_accuracy['Model']}"
    f" = {best_accuracy['Accuracy']:.4f}"
)


print(
    f"\nBest Macro F1:"
    f"\n{best_macro_f1['Model']}"
    f" = {best_macro_f1['Macro F1']:.4f}"
)


print(
    f"\nBest Weighted F1:"
    f"\n{best_weighted_f1['Model']}"
    f" = {best_weighted_f1['Weighted F1']:.4f}"
)


# ============================================================
# BEST TRANSFORMER
# ============================================================

if not transformer_comparison.empty:

    best_transformer = transformer_comparison.loc[
        transformer_comparison["Macro F1"].idxmax()
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST TRANSFORMER"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBest Transformer by Macro F1:"
        f"\n{best_transformer['Model']}"
        f" = {best_transformer['Macro F1']:.4f}"
    )


# ============================================================
# ACCURACY PLOT
# ============================================================

print("\nGenerating accuracy comparison plot...")


plt.figure(
    figsize=(12, 7)
)


plt.bar(
    comparison_sorted["Model"],
    comparison_sorted["Accuracy"]
)


plt.title(
    "Accuracy Comparison: Traditional ML vs XLM-R"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Accuracy"
)


plt.xticks(
    rotation=30,
    ha="right"
)


plt.ylim(
    0,
    1
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy_comparison.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# MACRO F1 PLOT
# ============================================================

print(
    "Generating Macro F1 comparison plot..."
)


plt.figure(
    figsize=(12, 7)
)


plt.bar(
    comparison_sorted["Model"],
    comparison_sorted["Macro F1"]
)


plt.title(
    "Macro F1 Comparison: Traditional ML vs XLM-R"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Macro F1"
)


plt.xticks(
    rotation=30,
    ha="right"
)


plt.ylim(
    0,
    1
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "macro_f1_comparison.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# WEIGHTED F1 PLOT
# ============================================================

print(
    "Generating Weighted F1 comparison plot..."
)


plt.figure(
    figsize=(12, 7)
)


plt.bar(
    comparison_sorted["Model"],
    comparison_sorted["Weighted F1"]
)


plt.title(
    "Weighted F1 Comparison: Traditional ML vs XLM-R"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Weighted F1"
)


plt.xticks(
    rotation=30,
    ha="right"
)


plt.ylim(
    0,
    1
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "weighted_f1_comparison.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# MACRO F1 - TRANSFORMER ONLY PLOT
# ============================================================

if not transformer_comparison.empty:

    print(
        "Generating transformer-only Macro F1 plot..."
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        transformer_comparison["Model"],
        transformer_comparison["Macro F1"]
    )

    plt.title(
        "XLM-R Transformer Comparison - Macro F1"
    )

    plt.xlabel(
        "Transformer Model"
    )

    plt.ylabel(
        "Macro F1"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.ylim(
        0,
        1
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "transformer_macro_f1_comparison.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("STEP 12 - MODEL COMPARISON COMPLETED")
print("=" * 70)


print("\nFiles generated:")

print(
    "- results/model_comparison/"
    "complete_model_comparison.csv"
)

print(
    "- results/model_comparison/"
    "traditional_ml_comparison.csv"
)

print(
    "- results/model_comparison/"
    "transformer_only_comparison.csv"
)

print(
    "- results/model_comparison/"
    "model_ranking.csv"
)

print(
    "- results/model_comparison/"
    "accuracy_comparison.png"
)

print(
    "- results/model_comparison/"
    "macro_f1_comparison.png"
)

print(
    "- results/model_comparison/"
    "weighted_f1_comparison.png"
)

print(
    "- results/model_comparison/"
    "transformer_macro_f1_comparison.png"
)
