import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from scipy.sparse import load_npz, hstack
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# STEP 7 - BASELINE MODEL EVALUATION
# ============================================================

print("=" * 60)
print("STEP 7 - BASELINE MODEL EVALUATION")
print("=" * 60)

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

X_TEST_WORD = "data/features/X_test_word.npz"
X_TEST_CHAR = "data/features/X_test_char.npz"
Y_TEST = "data/features/y_test.csv"

MODEL_DIR = "models/baseline_models"
RESULT_DIR = "results/baseline_evaluation"

os.makedirs(RESULT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2. Load test features
# ------------------------------------------------------------

print("\nLoading test features...")

X_test_word = load_npz(X_TEST_WORD)
X_test_char = load_npz(X_TEST_CHAR)

X_test = hstack(
    [X_test_word, X_test_char]
).tocsr()

y_test = pd.read_csv(Y_TEST)["label"]

print(f"Testing samples : {len(y_test)}")
print(f"Testing features: {X_test.shape}")

# ------------------------------------------------------------
# 3. Class information
# ------------------------------------------------------------

classes = sorted(y_test.unique())

print(f"\nNumber of classes: {len(classes)}")

print("\nClasses:")
for c in classes:
    print(f"- {c}")

# ------------------------------------------------------------
# 4. Load models
# ------------------------------------------------------------

model_files = {
    "Logistic Regression":
        "logistic_regression.pkl",

    "Linear SVM":
        "linear_svm.pkl",

    "Multinomial Naive Bayes":
        "multinomial_naive_bayes.pkl"
}

# ------------------------------------------------------------
# 5. Evaluate models
# ------------------------------------------------------------

results = []

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

for model_name, filename in model_files.items():

    print(f"\nEvaluating: {model_name}")

    model_path = os.path.join(MODEL_DIR, filename)

    model = joblib.load(model_path)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Macro Precision   : {macro_precision:.4f}")
    print(f"Macro Recall      : {macro_recall:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = os.path.join(
        RESULT_DIR,
        model_name.lower().replace(" ", "_")
        + "_classification_report.csv"
    )

    report_df.to_csv(report_path)

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=classes
    )

    plt.figure(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    disp.plot(
        xticks_rotation=45
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    cm_path = os.path.join(
        RESULT_DIR,
        model_name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "Actual_Label": y_test,
        "Predicted_Label": y_pred
    })

    prediction_path = os.path.join(
        RESULT_DIR,
        model_name.lower().replace(" ", "_")
        + "_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro_Precision": macro_precision,
        "Macro_Recall": macro_recall,
        "Macro_F1": macro_f1,
        "Weighted_Precision": weighted_precision,
        "Weighted_Recall": weighted_recall,
        "Weighted_F1": weighted_f1
    })

# ------------------------------------------------------------
# 6. Comparison
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
)

print("\n" + "=" * 60)
print("BASELINE MODEL EVALUATION SUMMARY")
print("=" * 60)

print(results_df.to_string(index=False))

# ------------------------------------------------------------
# 7. Save comparison
# ------------------------------------------------------------

comparison_path = os.path.join(
    RESULT_DIR,
    "baseline_evaluation_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

# ------------------------------------------------------------
# 8. Select best model using Macro F1
# ------------------------------------------------------------

best_model = results_df.iloc[0]

best_model_df = pd.DataFrame([best_model])

best_path = os.path.join(
    RESULT_DIR,
    "best_baseline_evaluation.csv"
)

best_model_df.to_csv(
    best_path,
    index=False
)

print("\n" + "=" * 60)
print("BEST BASELINE MODEL")
print("=" * 60)

print(
    f"Best Model : {best_model['Model']}"
)

print(
    f"Accuracy   : {best_model['Accuracy']:.4f}"
)

print(
    f"Macro F1   : {best_model['Macro_F1']:.4f}"
)

print("\n" + "=" * 60)
print("GENERATED FILES")
print("=" * 60)

print("\nResults:")
print("- results/baseline_evaluation/baseline_evaluation_comparison.csv")
print("- results/baseline_evaluation/best_baseline_evaluation.csv")

print("\nClassification reports:")
print("- One CSV for each baseline model")

print("\nConfusion matrices:")
print("- One PNG for each baseline model")

print("\nPrediction files:")
print("- One CSV for each baseline model")

print("\n" + "=" * 60)
print("BASELINE MODEL EVALUATION COMPLETED")
print("=" * 60)