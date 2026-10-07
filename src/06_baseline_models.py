import pandas as pd
import numpy as np
import os
import pickle
import time

from scipy.sparse import hstack, load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# STEP 6 - BASELINE MACHINE LEARNING MODELS
# ============================================================

print("=" * 60)
print("STEP 6 - BASELINE MACHINE LEARNING MODELS")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

FEATURE_DIR = "data/features"

RESULT_DIR = "results/baseline_models"

MODEL_DIR = "models/baseline_models"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD FEATURES
# ============================================================

print("\nLoading TF-IDF features...")

X_train_word = load_npz(
    os.path.join(FEATURE_DIR, "X_train_word.npz")
)

X_test_word = load_npz(
    os.path.join(FEATURE_DIR, "X_test_word.npz")
)

X_train_char = load_npz(
    os.path.join(FEATURE_DIR, "X_train_char.npz")
)

X_test_char = load_npz(
    os.path.join(FEATURE_DIR, "X_test_char.npz")
)


# ============================================================
# 3. LOAD LABELS
# ============================================================

y_train = pd.read_csv(
    os.path.join(FEATURE_DIR, "y_train.csv")
)["label"]

y_test = pd.read_csv(
    os.path.join(FEATURE_DIR, "y_test.csv")
)["label"]


print("\nTraining samples :", len(y_train))
print("Testing samples  :", len(y_test))


# ============================================================
# 4. CLASS INFORMATION
# ============================================================

classes = sorted(y_train.unique())

print("\nNumber of classes:", len(classes))

print("\nClasses:")
for c in classes:
    print("-", c)


# ============================================================
# 5. CLASS DISTRIBUTION
# ============================================================

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())


# ============================================================
# 6. COMBINE WORD + CHARACTER FEATURES
# ============================================================

print("\nCombining Word + Character TF-IDF features...")

X_train_combined = hstack([
    X_train_word,
    X_train_char
])

X_test_combined = hstack([
    X_test_word,
    X_test_char
])


print("\nCombined training shape:",
      X_train_combined.shape)

print("Combined testing shape :",
      X_test_combined.shape)


# ============================================================
# 7. MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB(
        alpha=0.5
    )
}


# ============================================================
# 8. TRAIN MODELS
# ============================================================

results = []

classification_reports = {}


print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)


for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    start_time = time.time()

    model.fit(
        X_train_combined,
        y_train
    )

    training_time = time.time() - start_time

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test_combined
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Macro Precision   : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall      : {macro_recall:.4f}"
    )

    print(
        f"Macro F1          : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1       : {weighted_f1:.4f}"
    )

    print(
        f"Training Time     : {training_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    report_path = os.path.join(
        RESULT_DIR,
        f"{safe_name}_classification_report.csv"
    )

    report_df.to_csv(report_path)

    classification_reports[model_name] = report_df

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "actual_label": y_test.values,
        "predicted_label": y_pred
    })

    prediction_path = os.path.join(
        RESULT_DIR,
        f"{safe_name}_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    # --------------------------------------------------------
    # Store overall results
    # --------------------------------------------------------

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Macro_Precision": macro_precision,

        "Macro_Recall": macro_recall,

        "Macro_F1": macro_f1,

        "Weighted_Precision": weighted_precision,

        "Weighted_Recall": weighted_recall,

        "Weighted_F1": weighted_f1,

        "Training_Time_Seconds": training_time
    })

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"{safe_name}.pkl"
    )

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(
        f"Model saved: {model_path}"
    )


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
)


print("\n" + "=" * 60)
print("BASELINE MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(index=False)
)


# ============================================================
# 10. SAVE COMPARISON
# ============================================================

comparison_path = os.path.join(
    RESULT_DIR,
    "baseline_model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 11. BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_macro_f1 = results_df.iloc[0]["Macro_F1"]


print("\n" + "=" * 60)
print("BEST BASELINE MODEL")
print("=" * 60)

print(
    f"Best Model : {best_model_name}"
)

print(
    f"Macro F1   : {best_macro_f1:.4f}"
)


# ============================================================
# 12. SAVE BEST MODEL INFORMATION
# ============================================================

best_model_info = pd.DataFrame({

    "Best_Model": [best_model_name],

    "Macro_F1": [best_macro_f1]

})

best_model_info.to_csv(
    os.path.join(
        RESULT_DIR,
        "best_baseline_model.csv"
    ),
    index=False
)


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("GENERATED FILES")
print("=" * 60)

print("\nResults:")
print(
    "- results/baseline_models/baseline_model_comparison.csv"
)

print(
    "- results/baseline_models/best_baseline_model.csv"
)

print("\nClassification reports:")
print(
    "- Logistic Regression classification report"
)

print(
    "- Linear SVM classification report"
)

print(
    "- Multinomial Naive Bayes classification report"
)

print("\nPrediction files:")
print(
    "- Logistic Regression predictions"
)

print(
    "- Linear SVM predictions"
)

print(
    "- Multinomial Naive Bayes predictions"
)

print("\nModels:")
print(
    "- models/baseline_models/logistic_regression.pkl"
)

print(
    "- models/baseline_models/linear_svm.pkl"
)

print(
    "- models/baseline_models/multinomial_naive_bayes.pkl"
)


print("\n" + "=" * 60)
print("BASELINE MODEL TRAINING COMPLETED")
print("=" * 60)