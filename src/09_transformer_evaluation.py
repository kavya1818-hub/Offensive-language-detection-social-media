import os
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# STEP 9 - TRANSFORMER EVALUATION
# ============================================================

print("=" * 60)
print("STEP 9 - XLM-R TRANSFORMER EVALUATION")
print("=" * 60)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/xlm_roberta"
TEST_FILE = "data/processed/test.csv"

OUTPUT_DIR = "results/transformer_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nCOMPUTING DEVICE")
print("-" * 60)

if torch.cuda.is_available():
    print("GPU available.")
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("GPU not available.")
    print("Using CPU.")

print("Device:", device)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"XLM-R model not found at: {MODEL_PATH}"
    )

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(
        f"Test dataset not found at: {TEST_FILE}"
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

test_df = pd.read_csv(TEST_FILE)

print("\nTesting records:", len(test_df))

print("\nAvailable columns:")
print(test_df.columns.tolist())


# ============================================================
# DETECT TEXT AND LABEL COLUMNS
# ============================================================

text_candidates = [
    "model_text",
    "normalized_text",
    "text",
    "tweet"
]

label_candidates = [
    "label",
    "target",
    "class"
]

text_column = None
label_column = None

for column in text_candidates:
    if column in test_df.columns:
        text_column = column
        break

for column in label_candidates:
    if column in test_df.columns:
        label_column = column
        break


if text_column is None:
    raise ValueError(
        "Could not find text column."
    )

if label_column is None:
    raise ValueError(
        "Could not find label column."
    )


print("\nText column :", text_column)
print("Label column:", label_column)


# ============================================================
# CLASS INFORMATION
# ============================================================

classes = sorted(
    test_df[label_column].unique()
)

print("\nNumber of classes:", len(classes))

print("\nClasses:")

for c in classes:
    print("-", c)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Tokenizer loaded successfully.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING TRAINED XLM-R MODEL")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# LABEL MAPPING
# ============================================================

print("\nModel label mapping:")

print(model.config.id2label)

# Convert dataset labels to model IDs
label2id = {}

for idx, class_name in enumerate(classes):
    label2id[class_name] = idx

print("\nDataset label mapping:")
print(label2id)


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("GENERATING XLM-R PREDICTIONS")
print("=" * 60)

texts = test_df[text_column].fillna("").astype(str).tolist()

true_labels = test_df[label_column].tolist()

predicted_labels = []
confidence_scores = []

batch_size = 16
max_length = 64

total = len(texts)

print("\nTesting samples:", total)
print("Batch size:", batch_size)
print("Maximum length:", max_length)

with torch.no_grad():

    for start in range(0, total, batch_size):

        end = min(
            start + batch_size,
            total
        )

        batch_texts = texts[start:end]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        outputs = model(**encoded)

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        confidences = torch.max(
            probabilities,
            dim=1
        ).values

        for prediction, confidence in zip(
            predictions.cpu().numpy(),
            confidences.cpu().numpy()
        ):

            prediction = int(prediction)

            # Use model's label mapping when available
            if prediction in model.config.id2label:

                predicted_label = (
                    model.config.id2label[prediction]
                )

                # Handle LABEL_0 / LABEL_1 style mappings
                if predicted_label.startswith("LABEL_"):

                    model_id = int(
                        predicted_label.split("_")[1]
                    )

                    predicted_label = classes[
                        model_id
                    ]

            else:
                predicted_label = classes[
                    prediction
                ]

            predicted_labels.append(
                predicted_label
            )

            confidence_scores.append(
                float(confidence)
            )

        completed = end

        if completed % 500 == 0 or completed == total:
            print(
                f"Processed {completed}/{total} samples"
            )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = test_df.copy()

results_df["true_label"] = true_labels
results_df["predicted_label"] = predicted_labels
results_df["confidence"] = confidence_scores

results_df["correct"] = (
    results_df["true_label"]
    ==
    results_df["predicted_label"]
)


# ============================================================
# METRICS
# ============================================================

print("\n" + "=" * 60)
print("XLM-R TEST RESULTS")
print("=" * 60)

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )
)

print(f"Accuracy           : {accuracy:.4f}")
print(f"Macro Precision    : {macro_precision:.4f}")
print(f"Macro Recall       : {macro_recall:.4f}")
print(f"Macro F1           : {macro_f1:.4f}")
print(f"Weighted Precision : {weighted_precision:.4f}")
print(f"Weighted Recall    : {weighted_recall:.4f}")
print(f"Weighted F1       : {weighted_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    true_labels,
    predicted_labels,
    labels=classes,
    target_names=classes,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

print(report_df)


report_path = os.path.join(
    OUTPUT_DIR,
    "xlm_roberta_classification_report.csv"
)

report_df.to_csv(report_path)

print("\nClassification report saved:")
print(report_path)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=classes
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual_{c}"
        for c in classes
    ],
    columns=[
        f"Predicted_{c}"
        for c in classes
    ]
)

print(cm_df)

cm_path = os.path.join(
    OUTPUT_DIR,
    "xlm_roberta_confusion_matrix.csv"
)

cm_df.to_csv(cm_path)

print("\nConfusion matrix saved:")
print(cm_path)


# ============================================================
# PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    OUTPUT_DIR,
    "xlm_roberta_predictions.csv"
)

results_df.to_csv(
    prediction_path,
    index=False
)

print("\nPrediction results saved:")
print(prediction_path)


# ============================================================
# ERROR ANALYSIS
# ============================================================

errors_df = results_df[
    results_df["correct"] == False
].copy()

print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

print("\nTotal samples :", len(results_df))
print("Correct       :", results_df["correct"].sum())
print("Incorrect     :", len(errors_df))

error_rate = (
    len(errors_df) /
    len(results_df)
)

print(f"Error rate    : {error_rate:.4f}")


error_path = os.path.join(
    OUTPUT_DIR,
    "xlm_roberta_misclassified.csv"
)

errors_df.to_csv(
    error_path,
    index=False
)

print("\nMisclassified samples saved:")
print(error_path)


# ============================================================
# CONFUSION PAIRS
# ============================================================

if len(errors_df) > 0:

    errors_df["error_type"] = (
        errors_df["true_label"]
        + " -> "
        + errors_df["predicted_label"]
    )

    error_pairs = (
        errors_df["error_type"]
        .value_counts()
        .reset_index()
    )

    error_pairs.columns = [
        "Error_Type",
        "Count"
    ]

    print("\nMost common errors:")
    print(error_pairs)

    error_pairs_path = os.path.join(
        OUTPUT_DIR,
        "xlm_roberta_error_pairs.csv"
    )

    error_pairs.to_csv(
        error_pairs_path,
        index=False
    )

    print("\nError pair analysis saved:")
    print(error_pairs_path)


# ============================================================
# CLASS-WISE SUMMARY
# ============================================================

class_summary = []

for class_name in classes:

    actual = (
        results_df["true_label"]
        == class_name
    ).sum()

    predicted = (
        results_df["predicted_label"]
        == class_name
    ).sum()

    correct = (
        (results_df["true_label"] == class_name)
        &
        (results_df["predicted_label"] == class_name)
    ).sum()

    recall = (
        correct / actual
        if actual > 0
        else 0
    )

    precision = (
        correct / predicted
        if predicted > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    class_summary.append({
        "Class": class_name,
        "Actual": actual,
        "Predicted": predicted,
        "Correct": correct,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


class_summary_df = pd.DataFrame(
    class_summary
)

print("\n" + "=" * 60)
print("CLASS-WISE PERFORMANCE")
print("=" * 60)

print(class_summary_df)

class_summary_path = os.path.join(
    OUTPUT_DIR,
    "xlm_roberta_class_wise_performance.csv"
)

class_summary_df.to_csv(
    class_summary_path,
    index=False
)

print("\nClass-wise results saved:")
print(class_summary_path)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame([{
    "Model": "XLM-R",
    "Accuracy": accuracy,
    "Macro_Precision": macro_precision,
    "Macro_Recall": macro_recall,
    "Macro_F1": macro_f1,
    "Weighted_Precision": weighted_precision,
    "Weighted_Recall": weighted_recall,
    "Weighted_F1": weighted_f1
}])

summary_path = os.path.join(
    OUTPUT_DIR,
    "xlm_roberta_evaluation_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print("\n" + "=" * 60)
print("STEP 9 - TRANSFORMER EVALUATION COMPLETED")
print("=" * 60)

print("\nFinal XLM-R Results:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Macro F1  : {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print("\nGenerated files:")
print(f"- {summary_path}")
print(f"- {report_path}")
print(f"- {cm_path}")
print(f"- {prediction_path}")
print(f"- {error_path}")
print(f"- {class_summary_path}")

print("\n" + "=" * 60)