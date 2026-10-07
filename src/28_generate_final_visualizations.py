"""
================================================================================
STEP 28 - FINAL VISUALIZATIONS - CORRECTED
================================================================================

Purpose:
    Generate final graphs and confusion matrices from the already completed
    Step 26 independent-test predictions.

IMPORTANT:
    - NO MODEL TRAINING
    - NO MODEL RETRAINING
    - NO DATASET MODIFICATION
    - NO CHANGES TO STEP 20
    - NO CHANGES TO STEP 24
    - NO CHANGES TO STEP 26

This corrected version:
    1. Handles numeric and string label columns.
    2. Correctly handles English and Overall prediction files.
    3. Verifies every confusion matrix against Step 26 sample counts.
    4. Removes the Matplotlib set_ticklabels warning.

================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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
    "step28_visualizations"
)

os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# 2. REQUIRED FILES
# =============================================================================

print("=" * 80)
print("STEP 28 - FINAL VISUALIZATIONS - CORRECTED")
print("=" * 80)

required_files = {
    "Step 26 summary":
        os.path.join(
            STEP26_DIR,
            "final_independent_test_summary.csv"
        ),

    "English predictions":
        os.path.join(
            STEP26_DIR,
            "dataset1_english_test_predictions.csv"
        ),

    "Tamil predictions":
        os.path.join(
            STEP26_DIR,
            "dataset2_tamil_test_predictions.csv"
        ),

    "Malayalam predictions":
        os.path.join(
            STEP26_DIR,
            "dataset2_malayalam_test_predictions.csv"
        ),

    "Kannada predictions":
        os.path.join(
            STEP26_DIR,
            "dataset2_kannada_test_predictions.csv"
        ),

    "Dravidian combined predictions":
        os.path.join(
            STEP26_DIR,
            "dataset2_dravidian_combined_test_predictions.csv"
        ),

    "Overall predictions":
        os.path.join(
            STEP26_DIR,
            "overall_independent_test_predictions.csv"
        )
}


for name, path in required_files.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{name}\n"
            f"{path}"
        )

    print(f"[FOUND] {name}")


print("\nAll required Step 26 files found.")


# =============================================================================
# 3. LOAD STEP 26 SUMMARY
# =============================================================================

summary = pd.read_csv(
    required_files["Step 26 summary"]
)

print("\nStep 26 summary loaded successfully.")


# =============================================================================
# 4. LABEL CONVERSION
# =============================================================================

def convert_label(value):
    """
    Convert possible label representations into:
        0 = Non-Offensive
        1 = Offensive

    Handles:
        0 / 1
        numeric strings
        Non_Offensive
        Non-Offensive
        Non Offensive
        Offensive
    """

    if pd.isna(value):
        return np.nan

    # Numeric
    if isinstance(value, (int, np.integer)):
        if value in [0, 1]:
            return int(value)

    if isinstance(value, (float, np.floating)):
        if value in [0.0, 1.0]:
            return int(value)

    text = str(value).strip().lower()

    # Numeric strings
    if text in ["0", "0.0"]:
        return 0

    if text in ["1", "1.0"]:
        return 1

    # Non-offensive variants
    non_offensive = {
        "non_offensive",
        "non-offensive",
        "non offensive",
        "nonoffensive",
        "not offensive",
        "not_offensive",
        "not-offensive",
        "0_non_offensive",
        "0 non_offensive"
    }

    if text in non_offensive:
        return 0

    # Offensive variants
    offensive = {
        "offensive",
        "1_offensive",
        "1 offensive"
    }

    if text in offensive:
        return 1

    # Unknown
    return np.nan


# =============================================================================
# 5. FIND ACTUAL AND PREDICTED COLUMNS
# =============================================================================

def find_prediction_columns(df):

    columns = list(df.columns)

    lower_map = {
        str(col).strip().lower(): col
        for col in columns
    }

    # Actual label candidates
    actual_candidates = [
        "label_id",
        "label",
        "actual_label",
        "actual",
        "true_label",
        "y_true"
    ]

    # Predicted label candidates
    predicted_candidates = [
        "predicted_label",
        "prediction",
        "pred_label",
        "predicted",
        "predicted_label_id",
        "y_pred"
    ]

    actual_col = None
    predicted_col = None

    for candidate in actual_candidates:

        if candidate in lower_map:

            actual_col = lower_map[candidate]
            break

    for candidate in predicted_candidates:

        if candidate in lower_map:

            predicted_col = lower_map[candidate]
            break

    if actual_col is None:

        raise ValueError(
            "\nCould not find actual-label column.\n"
            f"Available columns: {columns}"
        )

    if predicted_col is None:

        raise ValueError(
            "\nCould not find predicted-label column.\n"
            f"Available columns: {columns}"
        )

    return actual_col, predicted_col


# =============================================================================
# 6. CREATE CONFUSION MATRIX
# =============================================================================

def create_confusion_matrix(
    csv_path,
    title,
    output_name,
    expected_samples
):

    print("\n" + "-" * 80)
    print(f"Creating confusion matrix: {title}")
    print("-" * 80)

    df = pd.read_csv(csv_path)

    print("Columns:")
    print(list(df.columns))

    actual_col, predicted_col = find_prediction_columns(df)

    print("Actual column   :", actual_col)
    print("Predicted column:", predicted_col)

    # Convert labels
    y_true = df[actual_col].apply(
        convert_label
    )

    y_pred = df[predicted_col].apply(
        convert_label
    )

    # Keep only valid rows
    valid = (
        y_true.notna()
        &
        y_pred.notna()
    )

    y_true = y_true[valid].astype(int)
    y_pred = y_pred[valid].astype(int)

    sample_count = len(y_true)

    print("Valid samples   :", sample_count)
    print("Expected samples:", expected_samples)

    # IMPORTANT SAFETY CHECK
    if sample_count != expected_samples:

        raise ValueError(
            "\nCONFUSION MATRIX SAMPLE COUNT MISMATCH!\n"
            f"File: {csv_path}\n"
            f"Expected: {expected_samples}\n"
            f"Found: {sample_count}\n\n"
            "The visualization was NOT generated."
        )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:")
    print(cm)

    # Verify total
    if cm.sum() != expected_samples:

        raise ValueError(
            "\nCONFUSION MATRIX TOTAL DOES NOT MATCH EXPECTED SAMPLES."
        )

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(cm)

    ax.set_title(
        title,
        fontsize=14
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=11
    )

    ax.set_ylabel(
        "Actual Label",
        fontsize=11
    )

    labels = [
        "Non-Offensive",
        "Offensive"
    ]

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Cell values
    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14
            )

    fig.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUT_DIR,
        output_name
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_path)

    return cm


# =============================================================================
# 7. EXPECTED SAMPLE COUNTS FROM STEP 26
# =============================================================================

expected_counts = {
    "English": 315,
    "Tamil": 4146,
    "Malayalam": 851,
    "Kannada": 574,
    "Dravidian": 5571,
    "Overall": 5886
}


# =============================================================================
# 8. CONFUSION MATRICES
# =============================================================================

english_cm = create_confusion_matrix(
    required_files["English predictions"],
    "English Test - Confusion Matrix",
    "confusion_matrix_english.png",
    expected_counts["English"]
)

tamil_cm = create_confusion_matrix(
    required_files["Tamil predictions"],
    "Tamil Test - Confusion Matrix",
    "confusion_matrix_tamil.png",
    expected_counts["Tamil"]
)

malayalam_cm = create_confusion_matrix(
    required_files["Malayalam predictions"],
    "Malayalam Test - Confusion Matrix",
    "confusion_matrix_malayalam.png",
    expected_counts["Malayalam"]
)

kannada_cm = create_confusion_matrix(
    required_files["Kannada predictions"],
    "Kannada Test - Confusion Matrix",
    "confusion_matrix_kannada.png",
    expected_counts["Kannada"]
)

dravidian_cm = create_confusion_matrix(
    required_files["Dravidian combined predictions"],
    "Dravidian Combined Test - Confusion Matrix",
    "confusion_matrix_dravidian_combined.png",
    expected_counts["Dravidian"]
)

overall_cm = create_confusion_matrix(
    required_files["Overall predictions"],
    "Overall Independent Test - Confusion Matrix",
    "confusion_matrix_overall.png",
    expected_counts["Overall"]
)


# =============================================================================
# 9. LANGUAGE-WISE RESULTS
# =============================================================================

language_names = [
    "English",
    "Tamil",
    "Malayalam",
    "Kannada"
]

mapping = {
    "English": "DATASET 1 - ENGLISH TEST",
    "Tamil": "DATASET 2 - Tamil TEST",
    "Malayalam": "DATASET 2 - Malayalam TEST",
    "Kannada": "DATASET 2 - Kannada TEST"
}


language_rows = []

for language in language_names:

    match = summary[
        summary["Dataset / Language"].str.strip()
        ==
        mapping[language]
    ]

    if len(match) != 1:

        raise ValueError(
            f"Could not find unique result for {language}."
        )

    row = match.iloc[0]

    language_rows.append({
        "Language": language,
        "Samples": int(row["Samples"]),
        "Accuracy": float(row["Accuracy"]),
        "Macro-F1": float(row["Macro-F1"]),
        "Offensive Precision":
            float(row["Offensive Precision"]),
        "Offensive Recall":
            float(row["Offensive Recall"]),
        "Offensive F1":
            float(row["Offensive F1"])
    })


language_df = pd.DataFrame(
    language_rows
)


language_csv = os.path.join(
    OUT_DIR,
    "language_wise_visualization_metrics.csv"
)

language_df.to_csv(
    language_csv,
    index=False
)

print("\nSaved:", language_csv)


# =============================================================================
# 10. GENERIC BAR CHART FUNCTION
# =============================================================================

def create_metric_chart(
    x_labels,
    values,
    title,
    y_label,
    output_name
):

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    x = np.arange(
        len(x_labels)
    )

    bars = ax.bar(
        x,
        values
    )

    ax.set_title(
        title,
        fontsize=14
    )

    ax.set_xlabel(
        "Language"
    )

    ax.set_ylabel(
        y_label
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_yticks(
        np.arange(
            0,
            1.1,
            0.1
        )
    )

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    output_path = os.path.join(
        OUT_DIR,
        output_name
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_path)


# =============================================================================
# 11. ACCURACY
# =============================================================================

create_metric_chart(
    language_df["Language"].tolist(),
    language_df["Accuracy"].tolist(),
    "Accuracy by Language",
    "Accuracy",
    "accuracy_by_language.png"
)


# =============================================================================
# 12. MACRO-F1
# =============================================================================

create_metric_chart(
    language_df["Language"].tolist(),
    language_df["Macro-F1"].tolist(),
    "Macro-F1 by Language",
    "Macro-F1",
    "macro_f1_by_language.png"
)


# =============================================================================
# 13. OFFENSIVE F1
# =============================================================================

create_metric_chart(
    language_df["Language"].tolist(),
    language_df["Offensive F1"].tolist(),
    "Offensive-Class F1 by Language",
    "Offensive F1",
    "offensive_f1_by_language.png"
)


# =============================================================================
# 14. PRECISION / RECALL / F1
# =============================================================================

print("\nCreating Offensive precision-recall-F1 chart...")

x = np.arange(
    len(language_df)
)

width = 0.25

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    x - width,
    language_df["Offensive Precision"],
    width,
    label="Precision"
)

ax.bar(
    x,
    language_df["Offensive Recall"],
    width,
    label="Recall"
)

ax.bar(
    x + width,
    language_df["Offensive F1"],
    width,
    label="F1"
)

ax.set_title(
    "Offensive-Class Precision, Recall and F1",
    fontsize=14
)

ax.set_xlabel(
    "Language"
)

ax.set_ylabel(
    "Score"
)

ax.set_xticks(x)

ax.set_xticklabels(
    language_df["Language"]
)

ax.set_ylim(
    0,
    1.05
)

ax.set_yticks(
    np.arange(
        0,
        1.1,
        0.1
    )
)

ax.legend()

plt.tight_layout()

prf_path = os.path.join(
    OUT_DIR,
    "offensive_precision_recall_f1.png"
)

plt.savefig(
    prf_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", prf_path)


# =============================================================================
# 15. OVERALL FINAL METRICS
# =============================================================================

overall_match = summary[
    summary["Dataset / Language"].str.strip()
    ==
    "OVERALL INDEPENDENT TEST"
]

if len(overall_match) != 1:

    raise ValueError(
        "Overall independent-test result not found."
    )

overall_row = overall_match.iloc[0]


overall_metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Macro-F1",
        "Offensive Precision",
        "Offensive Recall",
        "Offensive F1"
    ],

    "Score": [
        float(overall_row["Accuracy"]),
        float(overall_row["Macro-F1"]),
        float(overall_row["Offensive Precision"]),
        float(overall_row["Offensive Recall"]),
        float(overall_row["Offensive F1"])
    ]
})


overall_metrics_csv = os.path.join(
    OUT_DIR,
    "overall_final_metrics.csv"
)

overall_metrics.to_csv(
    overall_metrics_csv,
    index=False
)

print("\nSaved:", overall_metrics_csv)


# =============================================================================
# 16. OVERALL PERFORMANCE CHART
# =============================================================================

print("\nCreating overall final performance chart...")

fig, ax = plt.subplots(
    figsize=(9, 6)
)

x = np.arange(
    len(overall_metrics)
)

bars = ax.bar(
    x,
    overall_metrics["Score"]
)

ax.set_title(
    "Final Overall Independent-Test Performance",
    fontsize=14
)

ax.set_ylabel(
    "Score"
)

ax.set_ylim(
    0,
    1.05
)

ax.set_xticks(x)

ax.set_xticklabels(
    overall_metrics["Metric"],
    rotation=20,
    ha="right"
)

ax.set_yticks(
    np.arange(
        0,
        1.1,
        0.1
    )
)

for bar, value in zip(
    bars,
    overall_metrics["Score"]
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

overall_chart_path = os.path.join(
    OUT_DIR,
    "overall_final_performance.png"
)

plt.savefig(
    overall_chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", overall_chart_path)


# =============================================================================
# 17. SAVE CONFUSION MATRICES AS CSV
# =============================================================================

confusion_matrices = {
    "english": english_cm,
    "tamil": tamil_cm,
    "malayalam": malayalam_cm,
    "kannada": kannada_cm,
    "dravidian_combined": dravidian_cm,
    "overall": overall_cm
}


for name, cm in confusion_matrices.items():

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

    path = os.path.join(
        OUT_DIR,
        f"confusion_matrix_{name}.csv"
    )

    cm_df.to_csv(path)

    print("Saved:", path)


# =============================================================================
# 18. SUMMARY TEXT
# =============================================================================

summary_txt = os.path.join(
    OUT_DIR,
    "step28_visualization_summary.txt"
)

with open(
    summary_txt,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 80 + "\n")
    f.write("STEP 28 - FINAL VISUALIZATION SUMMARY\n")
    f.write("=" * 80 + "\n\n")

    f.write(
        "Source: Step 26 independent-test prediction files.\n"
    )

    f.write(
        "No model was trained or modified.\n\n"
    )

    f.write("OVERALL INDEPENDENT TEST\n\n")

    f.write(
        f"Samples: {int(overall_row['Samples'])}\n"
    )

    f.write(
        f"Accuracy: {float(overall_row['Accuracy']):.4f}\n"
    )

    f.write(
        f"Macro-F1: {float(overall_row['Macro-F1']):.4f}\n"
    )

    f.write(
        f"Offensive Precision: "
        f"{float(overall_row['Offensive Precision']):.4f}\n"
    )

    f.write(
        f"Offensive Recall: "
        f"{float(overall_row['Offensive Recall']):.4f}\n"
    )

    f.write(
        f"Offensive F1: "
        f"{float(overall_row['Offensive F1']):.4f}\n\n"
    )

    f.write("LANGUAGE-WISE RESULTS\n\n")

    f.write(
        language_df.to_string(
            index=False
        )
    )

    f.write("\n")


# =============================================================================
# 19. FINAL VALIDATION
# =============================================================================

print("\n")
print("=" * 80)
print("FINAL VISUALIZATION VALIDATION")
print("=" * 80)

print(
    f"English confusion matrix total   : "
    f"{english_cm.sum()}"
)

print(
    f"Tamil confusion matrix total     : "
    f"{tamil_cm.sum()}"
)

print(
    f"Malayalam confusion matrix total : "
    f"{malayalam_cm.sum()}"
)

print(
    f"Kannada confusion matrix total   : "
    f"{kannada_cm.sum()}"
)

print(
    f"Dravidian confusion matrix total : "
    f"{dravidian_cm.sum()}"
)

print(
    f"Overall confusion matrix total   : "
    f"{overall_cm.sum()}"
)

assert english_cm.sum() == 315
assert tamil_cm.sum() == 4146
assert malayalam_cm.sum() == 851
assert kannada_cm.sum() == 574
assert dravidian_cm.sum() == 5571
assert overall_cm.sum() == 5886

print("\nALL CONFUSION MATRIX SAMPLE COUNTS VERIFIED.")


# =============================================================================
# 20. COMPLETE
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 28 COMPLETE - CORRECTED")
print("=" * 80)

print("\nVisualization directory:")
print(OUT_DIR)

print("\nNo model was trained.")
print("No model was modified.")
print("No Step 20/24/26 rerun was performed.")