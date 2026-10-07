import os
import json
import warnings

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "results",
    "multilingual_minority_aware",
    "final_model"
)

DATASET1 = os.path.join(
    BASE_DIR,
    "data",
    "train.csv"
)

DATASET2 = os.path.join(
    BASE_DIR,
    "data",
    "dravidian_clean",
    "dravidian_multilingual_clean.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "multilingual_minority_aware",
    "diagnostic"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LENGTH = 96

LABELS = {
    0: "Non_Offensive",
    1: "Offensive"
}

DEVICE = torch.device("cpu")

torch.set_num_threads(
    min(4, os.cpu_count() or 1)
)


# ============================================================
# PRINT
# ============================================================

def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_dataset1_label(value):

    value = str(value).strip()

    if value == "Non_Toxic":
        return 0

    if value in {
        "Homophobia_LGBTQphobia",
        "Misogyny_Sexism",
        "Racism_Xenophobia",
        "Harassment_Stalking",
        "Religious_Hate",
        "Profanity_Insult"
    }:
        return 1

    return None


def normalize_dataset2_label(value):

    value = str(value).strip().lower()

    if value in {
        "non_offensive",
        "non-offensive",
        "non offensive",
        "0"
    }:
        return 0

    if value in {
        "offensive",
        "1"
    }:
        return 1

    return None


# ============================================================
# LOAD MODEL
# ============================================================

section("LOADING STEP 20 XLM-R MODEL")

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"Step 20 model not found:\n{MODEL_DIR}"
    )

print("Model:")
print(MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.to(DEVICE)
model.eval()

print("Step 20 model loaded.")
print("Step 20 is NOT retrained.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_batch(texts):

    results = []

    for start in range(0, len(texts), 32):

        batch = texts[start:start + 32]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = model(**encoded)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

        results.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
            .tolist()
        )

        processed = min(
            start + len(batch),
            len(texts)
        )

        if (
            processed % 1000 == 0
            or processed == len(texts)
        ):
            print(
                f"Processed {processed}/{len(texts)}"
            )

    return np.array(results)


# ============================================================
# METRICS
# ============================================================

def evaluate_dataset(
    name,
    df,
    text_column,
    label_column,
    language_column=None
):

    section(f"{name} - OVERALL DIAGNOSTIC")

    texts = (
        df[text_column]
        .astype(str)
        .tolist()
    )

    y_true = (
        df[label_column]
        .astype(int)
        .to_numpy()
    )

    print(f"Samples: {len(df)}")

    offensive_probability = predict_batch(
        texts
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use 0.50 ONLY for diagnostic reporting.
    # We are NOT changing Step 24 calibration.
    # --------------------------------------------------------

    y_pred = (
        offensive_probability >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0
        )
    )

    print("\nAccuracy:", round(accuracy, 4))

    print(
        "\nNon_Offensive:"
    )
    print(
        f"Precision : {precision[0]:.4f}"
    )
    print(
        f"Recall    : {recall[0]:.4f}"
    )
    print(
        f"F1        : {f1[0]:.4f}"
    )

    print(
        "\nOffensive:"
    )
    print(
        f"Precision : {precision[1]:.4f}"
    )
    print(
        f"Recall    : {recall[1]:.4f}"
    )
    print(
        f"F1        : {f1[1]:.4f}"
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:")
    print(
        "                    Pred Non-Off   Pred Off"
    )
    print(
        f"Actual Non-Off      {cm[0,0]:10d}   {cm[0,1]:10d}"
    )
    print(
        f"Actual Off          {cm[1,0]:10d}   {cm[1,1]:10d}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    result = df.copy()

    result["offensive_probability"] = (
        offensive_probability
    )

    result["predicted_label"] = y_pred

    result["predicted_name"] = [
        LABELS[x]
        for x in y_pred
    ]

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{name.lower().replace(' ', '_')}_predictions.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        "\nPredictions saved:"
    )
    print(output_file)

    # --------------------------------------------------------
    # LANGUAGE-WISE
    # --------------------------------------------------------

    if language_column is not None:

        section(
            f"{name} - LANGUAGE-WISE DIAGNOSTIC"
        )

        for language in sorted(
            result[language_column]
            .dropna()
            .unique()
        ):

            subset = (
                result[
                    result[language_column]
                    == language
                ]
            )

            if len(subset) == 0:
                continue

            true_lang = (
                subset[label_column]
                .astype(int)
                .to_numpy()
            )

            pred_lang = (
                subset["predicted_label"]
                .astype(int)
                .to_numpy()
            )

            p, r, f, s = (
                precision_recall_fscore_support(
                    true_lang,
                    pred_lang,
                    labels=[0, 1],
                    zero_division=0
                )
            )

            acc = accuracy_score(
                true_lang,
                pred_lang
            )

            print(
                f"\n{language}"
            )
            print(
                f"Samples             : {len(subset)}"
            )
            print(
                f"Accuracy            : {acc:.4f}"
            )
            print(
                f"Offensive Precision  : {p[1]:.4f}"
            )
            print(
                f"Offensive Recall     : {r[1]:.4f}"
            )
            print(
                f"Offensive F1        : {f[1]:.4f}"
            )

    return result


# ============================================================
# DATASET 1
# ============================================================

section("LOADING DATASET 1 - ENGLISH")

if not os.path.exists(DATASET1):
    raise FileNotFoundError(DATASET1)

df1 = pd.read_csv(DATASET1)

print("Records:", len(df1))
print("Columns:", list(df1.columns))

df1["text_clean"] = (
    df1["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df1["label_id"] = (
    df1["label"]
    .apply(normalize_dataset1_label)
)

df1 = df1[
    (df1["text_clean"] != "")
    & df1["label_id"].notna()
].copy()

df1["label_id"] = (
    df1["label_id"]
    .astype(int)
)

print("\nDataset 1 distribution:")
print(
    df1["label_id"]
    .value_counts()
    .sort_index()
)

# ------------------------------------------------------------
# Sample equally from both classes for a fair diagnostic.
# ------------------------------------------------------------

n1 = min(
    2500,
    df1[
        df1["label_id"] == 0
    ].shape[0],
    df1[
        df1["label_id"] == 1
    ].shape[0]
)

diagnostic1 = pd.concat(
    [
        df1[
            df1["label_id"] == 0
        ].sample(
            n=n1,
            random_state=42
        ),

        df1[
            df1["label_id"] == 1
        ].sample(
            n=n1,
            random_state=42
        )
    ]
).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print(
    f"\nBalanced Dataset 1 diagnostic samples: "
    f"{len(diagnostic1)}"
)

english_results = evaluate_dataset(
    "Dataset1_English",
    diagnostic1,
    "text_clean",
    "label_id"
)


# ============================================================
# DATASET 2
# ============================================================

section("LOADING DATASET 2 - DRAVIDIAN")

if not os.path.exists(DATASET2):
    raise FileNotFoundError(DATASET2)

df2 = pd.read_csv(DATASET2)

print("Records:", len(df2))
print("Columns:", list(df2.columns))

df2["text_clean"] = (
    df2["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df2["label_id"] = (
    df2["mapped_label"]
    .apply(normalize_dataset2_label)
)

df2["language_clean"] = (
    df2["language"]
    .fillna("")
    .astype(str)
    .str.strip()
    .replace({
        "Tamil": "Tamil",
        "Malayalam": "Malayalam",
        "Kannada": "Kannada"
    })
)

df2 = df2[
    (df2["text_clean"] != "")
    & df2["label_id"].notna()
    & df2["language_clean"].isin(
        [
            "Tamil",
            "Malayalam",
            "Kannada"
        ]
    )
].copy()

df2["label_id"] = (
    df2["label_id"]
    .astype(int)
)

print("\nDataset 2 distribution:")
print(
    df2["label_id"]
    .value_counts()
    .sort_index()
)

print("\nLanguage distribution:")
print(
    df2["language_clean"]
    .value_counts()
)


# ============================================================
# BALANCED MULTILINGUAL DIAGNOSTIC
# ============================================================

diagnostic2_parts = []

for language in [
    "Tamil",
    "Malayalam",
    "Kannada"
]:

    lang_df = df2[
        df2["language_clean"]
        == language
    ]

    available_off = lang_df[
        lang_df["label_id"] == 1
    ]

    available_non = lang_df[
        lang_df["label_id"] == 0
    ]

    n = min(
        500,
        len(available_off),
        len(available_non)
    )

    if n == 0:
        print(
            f"\nWARNING: Cannot create balanced "
            f"diagnostic for {language}"
        )
        continue

    diagnostic2_parts.append(
        available_non.sample(
            n=n,
            random_state=42
        )
    )

    diagnostic2_parts.append(
        available_off.sample(
            n=n,
            random_state=42
        )
    )


diagnostic2 = pd.concat(
    diagnostic2_parts
).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print(
    "\nBalanced multilingual diagnostic samples:",
    len(diagnostic2)
)

dravidian_results = evaluate_dataset(
    "Dataset2_Dravidian",
    diagnostic2,
    "text_clean",
    "label_id",
    "language_clean"
)


# ============================================================
# MANUAL SENTENCE DIAGNOSTIC
# ============================================================

section("MANUAL SEMANTIC GENERALIZATION TEST")

manual_sentences = [

    # English
    (
        "English",
        "Direct insult",
        "You are an idiot.",
        1
    ),
    (
        "English",
        "Positive",
        "You are a wonderful person.",
        0
    ),
    (
        "English",
        "Contextual mention",
        "The character called an idiot in the story is funny.",
        0
    ),
    (
        "English",
        "Direct insult",
        "You are stupid and nobody likes you.",
        1
    ),
    (
        "English",
        "Neutral disagreement",
        "I disagree with your opinion.",
        0
    ),

    # Tamil
    (
        "Tamil",
        "Direct insult",
        "நீ ஒரு முட்டாள், உனக்கு எதுவுமே தெரியாது.",
        1
    ),
    (
        "Tamil",
        "Contextual mention",
        "இந்த கதையில் முட்டாள் என்ற கதாபாத்திரம் நகைச்சுவையாக காட்டப்படுகிறது.",
        0
    ),
    (
        "Tamil",
        "Positive",
        "இந்த படம் மிகவும் நன்றாக உள்ளது.",
        0
    ),

    # Malayalam
    (
        "Malayalam",
        "Direct insult",
        "നീ ഒരു മണ്ടനാണ്, നിനക്ക് ഒന്നും അറിയില്ല.",
        1
    ),
    (
        "Malayalam",
        "Contextual mention",
        "ഈ കഥയിലെ മണ്ടൻ എന്ന കഥാപാത്രം വളരെ തമാശക്കാരനാണ്.",
        0
    ),
    (
        "Malayalam",
        "Positive",
        "ഈ സിനിമ വളരെ നല്ലതാണ്.",
        0
    ),

    # Kannada
    (
        "Kannada",
        "Direct insult",
        "ನೀನು ಒಬ್ಬ ಮೂರ್ಖ, ನಿನಗೆ ಏನೂ ಗೊತ್ತಿಲ್ಲ.",
        1
    ),
    (
        "Kannada",
        "Contextual mention",
        "ಈ ಕಥೆಯಲ್ಲಿ ಮೂರ್ಖ ಎಂಬ ಪಾತ್ರವನ್ನು ಹಾಸ್ಯವಾಗಿ ತೋರಿಸಲಾಗಿದೆ.",
        0
    ),
    (
        "Kannada",
        "Positive",
        "ಈ ಸಿನಿಮಾ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.",
        0
    )
]

manual_texts = [
    x[2]
    for x in manual_sentences
]

manual_probs = predict_batch(
    manual_texts
)

manual_rows = []

for item, probability in zip(
    manual_sentences,
    manual_probs
):

    language, category, text, expected = item

    prediction = int(
        probability >= 0.50
    )

    manual_rows.append(
        {
            "language": language,
            "test_type": category,
            "text": text,
            "expected": LABELS[expected],
            "offensive_probability":
                round(float(probability), 6),
            "predicted":
                LABELS[prediction],
            "correct":
                prediction == expected
        }
    )

manual_df = pd.DataFrame(
    manual_rows
)

print(
    "\nManual test results:"
)

for _, row in manual_df.iterrows():

    print("\nLanguage :", row["language"])
    print("Type     :", row["test_type"])
    print("Text     :", row["text"])
    print("Expected :", row["expected"])
    print(
        "Offensive probability : "
        f"{row['offensive_probability']:.4f}"
    )
    print("Predicted :", row["predicted"])
    print(
        "Correct   : "
        f"{row['correct']}"
    )


manual_file = os.path.join(
    OUTPUT_DIR,
    "manual_semantic_tests.csv"
)

manual_df.to_csv(
    manual_file,
    index=False
)

print(
    "\nManual results saved:"
)
print(manual_file)


# ============================================================
# FINAL DIAGNOSIS
# ============================================================

section("FINAL DIAGNOSTIC SUMMARY")

english_off_recall = (
    precision_recall_fscore_support(
        diagnostic1["label_id"],
        english_results["predicted_label"],
        labels=[0, 1],
        zero_division=0
    )[1][1]
)

dravidian_off_recall = (
    precision_recall_fscore_support(
        diagnostic2["label_id"],
        dravidian_results["predicted_label"],
        labels=[0, 1],
        zero_division=0
    )[1][1]
)

manual_accuracy = (
    manual_df["correct"]
    .mean()
)

print(
    f"English Offensive Recall   : "
    f"{english_off_recall:.4f}"
)

print(
    f"Dravidian Offensive Recall : "
    f"{dravidian_off_recall:.4f}"
)

print(
    f"Manual Test Accuracy       : "
    f"{manual_accuracy:.4f}"
)

print("\nInterpretation:")

if english_off_recall < 0.70:

    print(
        "1. English offensive recall is weak."
    )

    print(
        "   The main problem is English generalization."
    )

elif manual_accuracy < 0.60:

    print(
        "1. Dataset performance is reasonable, "
        "but manual generalization is weak."
    )

    print(
        "   The classifier is not generalizing "
        "well to unseen sentence formulations."
    )

else:

    print(
        "1. Model generalization is reasonably strong."
    )

if dravidian_off_recall < 0.70:

    print(
        "2. One or more Dravidian languages "
        "have weak offensive detection."
    )

else:

    print(
        "2. Dravidian offensive detection is "
        "reasonably strong overall."
    )

print(
    "\nIMPORTANT:"
)

print(
    "Step 20 was NOT retrained."
)

print(
    "Step 24 v3 was NOT modified."
)

print(
    "No offensive words were hardcoded."
)

print(
    "No keyword rules were added."
)

print(
    "\nDiagnostic files:"
)

print(
    OUTPUT_DIR
)

print(
    "\nDo not change the model until these "
    "diagnostic results are inspected."
)
