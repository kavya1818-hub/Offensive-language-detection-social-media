"""
STEP 14 - FINAL RESULTS VISUALIZATION AND SUMMARY

Purpose:
    Create final publication/report-ready visualizations using
    already generated results from Steps 12 and 13.

IMPORTANT:
    This script does NOT train any model.
    It only reads CSV files and generates plots/tables.

Outputs:
    results/final_analysis/
        final_model_comparison.csv
        transformer_comparison.csv
        class_performance_comparison.csv
        final_summary.csv

        final_accuracy_comparison.png
        final_macro_f1_comparison.png
        final_weighted_f1_comparison.png
        transformer_metrics_comparison.png
        hate_speech_f1_comparison.png
        error_rate_comparison.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "results/final_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_bar_plot(
    df,
    model_col,
    value_col,
    title,
    ylabel,
    filename,
    rotation=25
):
    plt.figure(figsize=(11, 6))

    plt.bar(df[model_col], df[value_col])

    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel(ylabel)
    plt.xlabel("Model")

    plt.xticks(
        rotation=rotation,
        ha="right"
    )

    plt.ylim(0, 1.05)

    for i, value in enumerate(df[value_col]):
        plt.text(
            i,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# STEP 1 - LOAD MODEL COMPARISON
# ============================================================

print_header("STEP 14 - FINAL RESULTS VISUALIZATION")

comparison_file = (
    "results/model_comparison/complete_model_comparison.csv"
)

print("\nLoading final model comparison:")
print(comparison_file)

if not os.path.exists(comparison_file):
    raise FileNotFoundError(
        f"\nRequired file not found:\n{comparison_file}\n"
        "Run Step 12 first."
    )

comparison = pd.read_csv(comparison_file)

print("\nLoaded model comparison:")
print(comparison.to_string(index=False))


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

comparison.columns = [
    str(col).strip().replace("_", " ")
    for col in comparison.columns
]

# Restore expected names if needed
rename_map = {
    "Macro Precision": "Macro Precision",
    "Macro Recall": "Macro Recall",
    "Macro F1": "Macro F1",
    "Weighted F1": "Weighted F1",
    "Accuracy": "Accuracy",
    "Category": "Category",
    "Model": "Model"
}

comparison = comparison.rename(columns=rename_map)


# ============================================================
# SAVE CLEAN FINAL COMPARISON
# ============================================================

final_comparison_file = os.path.join(
    OUTPUT_DIR,
    "final_model_comparison.csv"
)

comparison.to_csv(
    final_comparison_file,
    index=False
)

print("\nFinal comparison saved:")
print(final_comparison_file)


# ============================================================
# STEP 2 - TRADITIONAL VS TRANSFORMER
# ============================================================

print_header("TRADITIONAL ML VS TRANSFORMER")

print(
    comparison[
        [
            "Category",
            "Model",
            "Accuracy",
            "Macro F1",
            "Weighted F1"
        ]
    ].to_string(index=False)
)


# ============================================================
# FINAL ACCURACY
# ============================================================

save_bar_plot(
    comparison,
    "Model",
    "Accuracy",
    "Accuracy Comparison - Traditional ML vs Transformers",
    "Accuracy",
    "final_accuracy_comparison.png"
)

print("\nAccuracy plot generated.")


# ============================================================
# FINAL MACRO F1
# ============================================================

save_bar_plot(
    comparison,
    "Model",
    "Macro F1",
    "Macro F1 Comparison - Traditional ML vs Transformers",
    "Macro F1",
    "final_macro_f1_comparison.png"
)

print("Macro F1 plot generated.")


# ============================================================
# FINAL WEIGHTED F1
# ============================================================

save_bar_plot(
    comparison,
    "Model",
    "Weighted F1",
    "Weighted F1 Comparison - Traditional ML vs Transformers",
    "Weighted F1",
    "final_weighted_f1_comparison.png"
)

print("Weighted F1 plot generated.")


# ============================================================
# STEP 3 - TRANSFORMER ONLY
# ============================================================

print_header("TRANSFORMER-ONLY ANALYSIS")

transformer_df = comparison[
    comparison["Category"].astype(str).str.contains(
        "Transformer",
        case=False,
        na=False
    )
].copy()

print(
    transformer_df[
        [
            "Model",
            "Accuracy",
            "Macro F1",
            "Weighted F1"
        ]
    ].to_string(index=False)
)

transformer_file = os.path.join(
    OUTPUT_DIR,
    "transformer_comparison.csv"
)

transformer_df.to_csv(
    transformer_file,
    index=False
)

print("\nTransformer comparison saved:")
print(transformer_file)


# ============================================================
# TRANSFORMER METRICS PLOT
# ============================================================

metrics = [
    "Accuracy",
    "Macro F1",
    "Weighted F1"
]

x = range(len(transformer_df))
width = 0.25

plt.figure(figsize=(11, 6))

for i, metric in enumerate(metrics):
    values = transformer_df[metric].values

    positions = [
        p + (i - 1) * width
        for p in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )

    for pos, value in zip(positions, values):
        plt.text(
            pos,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            fontsize=8
        )

plt.xticks(
    list(x),
    transformer_df["Model"],
    rotation=20,
    ha="right"
)

plt.ylabel("Score")
plt.title(
    "XLM-R Transformer Performance Comparison",
    fontsize=14,
    fontweight="bold"
)

plt.ylim(0, 1.05)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "transformer_metrics_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Transformer metrics plot generated.")


# ============================================================
# STEP 4 - CLASS-WISE PERFORMANCE
# ============================================================

print_header("CLASS-WISE PERFORMANCE ANALYSIS")

class_files = {
    "XLM-R Baseline":
        "results/error_analysis/xlmr_baseline_class_performance.csv",

    "XLM-R Balanced":
        "results/error_analysis/xlmr_balanced_class_performance.csv",

    "XLM-R Weighted":
        "results/error_analysis/xlmr_weighted_class_performance.csv"
}

class_results = []

for model_name, file_path in class_files.items():

    print(f"\nLoading {model_name}:")
    print(file_path)

    if not os.path.exists(file_path):
        print("WARNING: File not found. Skipping.")
        continue

    df = pd.read_csv(file_path)

    # Keep only useful columns
    required = [
        "Class",
        "Precision",
        "Recall",
        "F1"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        print(
            f"WARNING: Missing columns {missing}. Skipping."
        )
        continue

    df = df[required].copy()

    df["Model"] = model_name

    class_results.append(df)


if class_results:

    class_performance = pd.concat(
        class_results,
        ignore_index=True
    )

    # Put Model first
    class_performance = class_performance[
        [
            "Model",
            "Class",
            "Precision",
            "Recall",
            "F1"
        ]
    ]

    class_file = os.path.join(
        OUTPUT_DIR,
        "class_performance_comparison.csv"
    )

    class_performance.to_csv(
        class_file,
        index=False
    )

    print("\nCombined class-wise performance:")
    print(
        class_performance.to_string(index=False)
    )

    print("\nClass-wise comparison saved:")
    print(class_file)

else:

    print(
        "\nNo class-wise performance files were available."
    )


# ============================================================
# STEP 5 - HATE SPEECH PERFORMANCE
# ============================================================

print_header("HATE SPEECH PERFORMANCE")

hate_df = class_performance[
    class_performance["Class"].astype(str).str.lower()
    == "hate_speech"
].copy()

if not hate_df.empty:

    print(
        hate_df[
            [
                "Model",
                "Precision",
                "Recall",
                "F1"
            ]
        ].to_string(index=False)
    )

    save_bar_plot(
        hate_df,
        "Model",
        "F1",
        "Hate Speech F1 Comparison",
        "F1 Score",
        "hate_speech_f1_comparison.png"
    )

    print(
        "\nHate Speech F1 plot generated."
    )


# ============================================================
# STEP 6 - ERROR RATE
# ============================================================

print_header("ERROR RATE ANALYSIS")

error_file = (
    "results/error_analysis/error_summary.csv"
)

if os.path.exists(error_file):

    error_df = pd.read_csv(error_file)

    print("\nError summary:")
    print(
        error_df.to_string(index=False)
    )

    # Save copy
    error_output = os.path.join(
        OUTPUT_DIR,
        "error_summary.csv"
    )

    error_df.to_csv(
        error_output,
        index=False
    )

    # Detect error-rate column
    error_column = None

    for col in error_df.columns:
        if str(col).lower() == "error_rate":
            error_column = col
            break

    if error_column:

        save_bar_plot(
            error_df,
            "Model",
            error_column,
            "Model Error Rate Comparison",
            "Error Rate",
            "error_rate_comparison.png"
        )

        print(
            "\nError-rate plot generated."
        )

else:

    print(
        "\nWARNING: error_summary.csv not found."
    )


# ============================================================
# STEP 7 - FINAL SUMMARY
# ============================================================

print_header("FINAL MODEL SUMMARY")


best_accuracy = comparison.loc[
    comparison["Accuracy"].idxmax()
]

best_macro_f1 = comparison.loc[
    comparison["Macro F1"].idxmax()
]

best_weighted_f1 = comparison.loc[
    comparison["Weighted F1"].idxmax()
]


# Transformer-only best
if not transformer_df.empty:

    best_transformer = transformer_df.loc[
        transformer_df["Macro F1"].idxmax()
    ]

else:

    best_transformer = None


summary_rows = [
    {
        "Metric": "Best Accuracy",
        "Model": best_accuracy["Model"],
        "Category": best_accuracy["Category"],
        "Score": best_accuracy["Accuracy"]
    },
    {
        "Metric": "Best Macro F1",
        "Model": best_macro_f1["Model"],
        "Category": best_macro_f1["Category"],
        "Score": best_macro_f1["Macro F1"]
    },
    {
        "Metric": "Best Weighted F1",
        "Model": best_weighted_f1["Model"],
        "Category": best_weighted_f1["Category"],
        "Score": best_weighted_f1["Weighted F1"]
    }
]

if best_transformer is not None:

    summary_rows.append(
        {
            "Metric": "Best Transformer Macro F1",
            "Model": best_transformer["Model"],
            "Category": "Transformer",
            "Score": best_transformer["Macro F1"]
        }
    )


summary_df = pd.DataFrame(summary_rows)

summary_file = os.path.join(
    OUTPUT_DIR,
    "final_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

print(
    summary_df.to_string(index=False)
)

print("\nFinal summary saved:")
print(summary_file)


# ============================================================
# STEP 8 - IMPORTANT INTERPRETATION
# ============================================================

print_header("FINAL INTERPRETATION")

print(
    f"""
Best overall Accuracy:
{best_accuracy['Model']} = {best_accuracy['Accuracy']:.4f}

Best overall Macro F1:
{best_macro_f1['Model']} = {best_macro_f1['Macro F1']:.4f}

Best overall Weighted F1:
{best_weighted_f1['Model']} = {best_weighted_f1['Weighted F1']:.4f}
"""
)

if best_transformer is not None:

    print(
        f"""
Best Transformer by Macro F1:
{best_transformer['Model']} = {best_transformer['Macro F1']:.4f}
"""
    )

print(
    """
IMPORTANT PROJECT CONCLUSION:

Macro F1 is particularly important for this project because
the Hate Speech class is much smaller than Offensive Language.

Therefore, accuracy alone should NOT be used to claim that
a model is better.

The final analysis should consider:

1. Accuracy
2. Macro Precision
3. Macro Recall
4. Macro F1
5. Weighted F1
6. Hate Speech Recall
7. Hate Speech F1
8. Error rate
"""
)


# ============================================================
# COMPLETION
# ============================================================

print_header("STEP 14 - FINAL VISUALIZATION COMPLETED")

print(
    """
Generated files:

results/final_analysis/
    final_model_comparison.csv
    transformer_comparison.csv
    class_performance_comparison.csv
    final_summary.csv
    error_summary.csv

    final_accuracy_comparison.png
    final_macro_f1_comparison.png
    final_weighted_f1_comparison.png
    transformer_metrics_comparison.png
    hate_speech_f1_comparison.png
    error_rate_comparison.png

No model training was performed.
"""
)