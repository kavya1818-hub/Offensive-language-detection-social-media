# ==============================================================================
# STEP 32 - EVASION-AWARE MULTILINGUAL OFFENSIVE LANGUAGE DETECTION
# ==============================================================================
#
# Uses the EXISTING Step 24 model.
#
# Step 20 : FROZEN
# Step 24 : FROZEN
# Step 32 : INFERENCE-TIME EVASION CHECK ONLY
#
# No retraining
# No new classifier
# No keyword rules
# No offensive dictionary
# No manual sentence rules
#
# Architecture:
#
# Social Media Text
#       ↓
# Preprocessing
#       ↓
# Existing Step 24 XLM-R + Semantic Hybrid
#       ↓
# Initial probability
#       ↓
# Evasion Check
#       ↓
# Generic normalized variants
#       ↓
# SAME FROZEN STEP 24 MODEL
#       ↓
# Risk consistency
#       ↓
# Stable prediction / Possible Evasion
#
# ==============================================================================

import os
import re
import json
import unicodedata
import warnings

import numpy as np
import pandas as pd
import torch
import joblib

from transformers import AutoTokenizer, AutoModelForSequenceClassification


warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

CPU_THREADS = min(4, os.cpu_count() or 4)
MAX_LENGTH = 96
BATCH_SIZE = 8

torch.set_num_threads(CPU_THREADS)

try:
    torch.set_num_interop_threads(
        max(1, min(2, CPU_THREADS))
    )
except RuntimeError:
    pass

DEVICE = torch.device("cpu")


# ==============================================================================
# PATHS
# ==============================================================================

BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ROOT = os.path.join(
    BASE,
    "results",
    "multilingual_cross_dataset_final"
)

# EXISTING STEP 24
STEP24_ROOT = os.path.join(
    ROOT,
    "step24_semantic_final"
)

MODEL_DIR = os.path.join(
    ROOT,
    "final_model"
)

SEM_FILE = os.path.join(
    STEP24_ROOT,
    "semantic_model",
    "semantic_classifier.joblib"
)

CAL_FILE = os.path.join(
    STEP24_ROOT,
    "hybrid_calibration.json"
)

CFG_FILE = os.path.join(
    STEP24_ROOT,
    "step24_config.json"
)

# Step 32 output only
OUT_DIR = os.path.join(
    ROOT,
    "step32_evasion_aware"
)

CSV_FILE = os.path.join(
    OUT_DIR,
    "evasion_demo_results.csv"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ==============================================================================
# LABELS
# ==============================================================================

LABELS = {
    0: "Non_Offensive",
    1: "Offensive"
}


# ==============================================================================
# HEADER
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 32 - EVASION-AWARE MULTILINGUAL OFFENSIVE LANGUAGE DETECTION")
print("=" * 80)

print("Step 20 : FROZEN")
print("Step 24 : FROZEN")
print("Step 32 : INFERENCE-TIME EVASION CHECK")
print("Retraining : NONE")
print("Keyword rules : NONE")
print("Dictionary : NONE")
print("Manual sentence rules : NONE")
print("=" * 80)


# ==============================================================================
# STEP 1 - LOAD EXACT STEP 24 ARTIFACTS
# ==============================================================================

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"Step 20 model not found:\n{MODEL_DIR}"
    )

if not os.path.exists(SEM_FILE):
    raise FileNotFoundError(
        f"Step 24 semantic classifier not found:\n{SEM_FILE}"
    )

if not os.path.exists(CAL_FILE):
    raise FileNotFoundError(
        f"Step 24 calibration file not found:\n{CAL_FILE}"
    )

if not os.path.exists(CFG_FILE):
    raise FileNotFoundError(
        f"Step 24 configuration not found:\n{CFG_FILE}"
    )


print("\n[1/4] Loading existing Step 24 artifacts...")


# Same tokenizer/model used by Step 24
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
).to(DEVICE)

model.eval()


# Existing Step 24 semantic classifier
semantic_classifier = joblib.load(
    SEM_FILE
)


# Existing Step 24 calibration
with open(
    CAL_FILE,
    "r",
    encoding="utf-8"
) as f:
    calibration = json.load(f)


# Existing Step 24 configuration
with open(
    CFG_FILE,
    "r",
    encoding="utf-8"
) as f:
    step24_config = json.load(f)


THRESHOLD = float(
    calibration["threshold"]
)

XLMR_WEIGHT = float(
    calibration["xlmr_weight"]
)

SEMANTIC_WEIGHT = float(
    calibration["semantic_weight"]
)


print("Frozen Step 20 XLM-R loaded.")
print("Frozen Step 24 semantic classifier loaded.")

print(
    f"Step 24 threshold       : {THRESHOLD:.4f}"
)

print(
    f"Step 24 XLM-R weight    : {XLMR_WEIGHT:.4f}"
)

print(
    f"Step 24 semantic weight : {SEMANTIC_WEIGHT:.4f}"
)

print("No model retraining performed.")


# ==============================================================================
# STEP 2 - SAME TEXT NORMALIZATION AS STEP 24
# ==============================================================================

def textnorm(text):

    if pd.isna(text):
        return ""

    return " ".join(
        str(text).strip().split()
    )


# ==============================================================================
# STEP 3 - XLM-R PROBABILITY
# ==============================================================================

def xlmr_probability(text):

    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**encoded)

        probability = torch.softmax(
            outputs.logits,
            dim=-1
        )[0, 1].item()

    return float(probability)


# ==============================================================================
# STEP 4 - EXACT STEP 24 SEMANTIC EMBEDDING
# ==============================================================================

def semantic_embedding(text):

    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    with torch.no_grad():

        hidden = model.roberta(
            **encoded
        ).last_hidden_state

    mask = (
        encoded["attention_mask"]
        .unsqueeze(-1)
        .float()
    )

    masked_hidden = hidden * mask

    mean_pool = (
        masked_hidden.sum(dim=1)
        /
        mask.sum(dim=1).clamp(min=1e-9)
    )

    max_pool = hidden.masked_fill(
        mask == 0,
        torch.finfo(hidden.dtype).min
    ).max(dim=1).values

    embedding = torch.cat(
        [mean_pool, max_pool],
        dim=1
    )

    return embedding.numpy().astype(
        np.float32
    )


# ==============================================================================
# STEP 5 - EXACT STEP 24 HYBRID PREDICTION
# ==============================================================================

def step24_predict(text):

    """
    This reproduces the Step 24 real-time prediction:

        XLM-R probability
                 +
        Semantic probability
                 ↓
        Saved calibrated hybrid
                 ↓
        Saved threshold
    """

    text = textnorm(text)

    # XLM-R probability
    xlmr_prob = xlmr_probability(
        text
    )

    # Semantic probability
    embedding = semantic_embedding(
        text
    )

    semantic_prob = float(
        semantic_classifier.predict_proba(
            embedding
        )[0, 1]
    )

    # EXACT Step 24 hybrid
    hybrid_prob = (
        XLMR_WEIGHT * xlmr_prob
        +
        SEMANTIC_WEIGHT * semantic_prob
    )

    prediction = int(
        hybrid_prob >= THRESHOLD
    )

    confidence = (
        hybrid_prob
        if prediction == 1
        else 1.0 - hybrid_prob
    )

    return {
        "text": text,
        "xlmr_probability": xlmr_prob,
        "semantic_probability": semantic_prob,
        "hybrid_probability": hybrid_prob,
        "prediction": prediction,
        "label": LABELS[prediction],
        "confidence": confidence
    }


# ==============================================================================
# STEP 6 - EVASION TRANSFORMATIONS
# ==============================================================================

def remove_zero_width(text):

    return re.sub(
        r"[\u200B-\u200D\uFEFF]",
        "",
        text
    )


def normalize_unicode(text):

    return unicodedata.normalize(
        "NFKC",
        text
    )


def normalize_spaces(text):

    # Collapse repeated whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove spaces around punctuation
    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    return text.strip()


def normalize_repeated_punctuation(text):

    return re.sub(
        r"([!?.,])\1{1,}",
        r"\1",
        text
    )


def normalize_repeated_characters(text):

    # Example:
    # iiiiidiooot
    # → idi? No vocabulary is used.
    #
    # Generic structural normalization:
    # three or more consecutive identical characters
    # are reduced to two.

    return re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        text
    )


def normalize_character_spacing(text):


    return re.sub(
        r"(?<!\w)([A-Za-z])(?:\s+([A-Za-z])){2,}(?!\w)",
        lambda m: re.sub(
            r"\s+",
            "",
            m.group(0)
        ),
        text
    )


def normalize_leetspeak(text):

    # Generic character normalization.
    # This does NOT contain offensive words.

    translation = str.maketrans({
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t"
    })

    return text.translate(
        translation
    )

# ==============================================================================
# STEP 7 - SENTENCE-LEVEL EVASION VARIANTS
# ==============================================================================

def generate_variants(text):

    """
    Generate complete-sentence variants.

    The complete social-media sentence is preserved.
    Only character-level obfuscation is introduced inside
    the sentence.

    No offensive dictionary or keyword list is used.
    """

    variants = []

    # --------------------------------------------------------------------------
    # Variant 1: Insert spaces inside a selected word
    # --------------------------------------------------------------------------

    words = text.split()

    for i, word in enumerate(words):

        # Keep punctuation attached to the word.
        core = re.sub(
            r"^[^\w]+|[^\w]+$",
            "",
            word,
            flags=re.UNICODE
        )

        if len(core) < 5:
            continue

        start = word.find(core)

        # Split the word roughly in the middle.
        mid = len(core) // 2

        if mid <= 1 or mid >= len(core) - 1:
            continue

        obfuscated = (
            core[:mid]
            + " "
            + core[mid:]
        )

        new_word = (
            word[:start]
            + obfuscated
            + word[start + len(core):]
        )

        new_words = words.copy()
        new_words[i] = new_word

        variant = " ".join(new_words)

        if variant != text:

            variants.append(
                (
                    "Internal character spacing",
                    variant
                )
            )


    # --------------------------------------------------------------------------
    # Variant 2: Insert punctuation inside a word
    # --------------------------------------------------------------------------

    for i, word in enumerate(words):

        core = re.sub(
            r"^[^\w]+|[^\w]+$",
            "",
            word,
            flags=re.UNICODE
        )

        if len(core) < 5:
            continue

        start = word.find(core)
        mid = len(core) // 2

        if mid <= 1 or mid >= len(core) - 1:
            continue

        obfuscated = (
            core[:mid]
            + "."
            + core[mid:]
        )

        new_word = (
            word[:start]
            + obfuscated
            + word[start + len(core):]
        )

        new_words = words.copy()
        new_words[i] = new_word

        variant = " ".join(new_words)

        if variant != text:

            variants.append(
                (
                    "Internal punctuation",
                    variant
                )
            )


    # --------------------------------------------------------------------------
    # Variant 3: Repeated-character variation
    # --------------------------------------------------------------------------

    for i, word in enumerate(words):

        core = re.sub(
            r"^[^\w]+|[^\w]+$",
            "",
            word,
            flags=re.UNICODE
        )

        if len(core) < 5:
            continue

        start = word.find(core)

        # Repeat one internal character.
        pos = len(core) // 2

        if pos >= len(core):
            continue

        obfuscated = (
            core[:pos]
            + core[pos] * 3
            + core[pos + 1:]
        )

        new_word = (
            word[:start]
            + obfuscated
            + word[start + len(core):]
        )

        new_words = words.copy()
        new_words[i] = new_word

        variant = " ".join(new_words)

        if variant != text:

            variants.append(
                (
                    "Repeated-character variation",
                    variant
                )
            )


    # --------------------------------------------------------------------------
    # Variant 4: Zero-width character insertion
    # --------------------------------------------------------------------------

    zero_width = "\u200b"

    for i, word in enumerate(words):

        core = re.sub(
            r"^[^\w]+|[^\w]+$",
            "",
            word,
            flags=re.UNICODE
        )

        if len(core) < 5:
            continue

        start = word.find(core)
        pos = len(core) // 2

        obfuscated = (
            core[:pos]
            + zero_width
            + core[pos:]
        )

        new_word = (
            word[:start]
            + obfuscated
            + word[start + len(core):]
        )

        new_words = words.copy()
        new_words[i] = new_word

        variant = " ".join(new_words)

        if variant != text:

            variants.append(
                (
                    "Zero-width insertion",
                    variant
                )
            )


    # --------------------------------------------------------------------------
    # Remove duplicate variants
    # --------------------------------------------------------------------------

    unique = []
    seen = set()

    for name, variant in variants:

        if variant not in seen:

            seen.add(variant)

            unique.append(
                (
                    name,
                    variant
                )
            )

    return unique
# ==============================================================================
# STEP 8 - EVASION RISK ANALYSIS
# ==============================================================================

def evasion_analysis(text):

    original = step24_predict(
        text
    )

    variants = generate_variants(
        text
    )

    results = []

    for name, variant in variants:

        prediction = step24_predict(
            variant
        )

        probability_change = (
            prediction["hybrid_probability"]
            -
            original["hybrid_probability"]
        )

        results.append({

            "transformation": name,

            "variant": variant,

            "hybrid_probability":
                prediction["hybrid_probability"],

            "prediction":
                prediction["label"],

            "probability_change":
                probability_change
        })


    # --------------------------------------------------------------------------
    # Risk consistency
    # --------------------------------------------------------------------------

    evasion_detected = False

    reason = "Stable prediction"

    if results:

        # Any transformed version becoming Offensive
        # when the original was Non-Offensive
        # is considered a potential evasion signal.

        if original["prediction"] == 0:

            offensive_variants = [
                r for r in results
                if r["prediction"] == "Offensive"
            ]

            if offensive_variants:

                max_variant = max(
                    offensive_variants,
                    key=lambda x:
                    x["hybrid_probability"]
                )

                probability_increase = (
                    max_variant["hybrid_probability"]
                    -
                    original["hybrid_probability"]
                )

                # Require meaningful instability.
                if probability_increase >= 0.20:

                    evasion_detected = True

                    reason = (
                        "A normalized variant changed "
                        "the prediction to Offensive "
                        "with a significant probability increase."
                    )


    if evasion_detected:

        final_decision = (
            "POSSIBLE EVASION - REVIEW"
        )

    else:

        final_decision = (
            original["label"]
        )


    return {
        "original": original,
        "variants": results,
        "evasion_detected": evasion_detected,
        "reason": reason,
        "final_decision": final_decision
    }


# ==============================================================================
# STEP 9 - DISPLAY RESULT
# ==============================================================================

def display_result(
    text,
    analysis
):

    original = analysis["original"]

    print("\n")
    print("=" * 80)
    print("EVASION-AWARE MODERATION RESULT")
    print("=" * 80)

    print("\nINPUT")
    print("-" * 80)
    print(text)

    print("\nORIGINAL STEP 24 PREDICTION")
    print("-" * 80)

    print(
        f"XLM-R probability       : "
        f"{original['xlmr_probability'] * 100:.2f}%"
    )

    print(
        f"Semantic probability    : "
        f"{original['semantic_probability'] * 100:.2f}%"
    )

    print(
        f"Hybrid probability      : "
        f"{original['hybrid_probability'] * 100:.2f}%"
    )

    print(
        f"Original prediction     : "
        f"{original['label']}"
    )

    print("\nEVASION CHECK")
    print("-" * 80)

    if not analysis["variants"]:

        print(
            "No meaningful transformation generated."
        )

    else:

        print(
            f"{'Transformation':32}"
            f"{'Hybrid':12}"
            f"{'Change':12}"
            f"{'Prediction'}"
        )
        print("-" * 80)
        for r in analysis["variants"]:
            print(
                f"{r['transformation'][:32]:32}"
                f"{r['hybrid_probability'] * 100:>7.2f}%"
                f"{r['probability_change'] * 100:>+9.2f}%"
                f"    {r['prediction']}"
            )
            print(
                f"  Variant: {r['variant']}"
            )

    print("\nRISK CONSISTENCY")
    print("-" * 80)

    print(
        "Evasion detected : "
        + (
            "YES"
            if analysis["evasion_detected"]
            else "NO"
        )
    )

    print(
        "Reason           : "
        + analysis["reason"]
    )

    print("\nFINAL MODERATION DECISION")
    print("-" * 80)

    print(
        f"FINAL DECISION   : "
        f"{analysis['final_decision']}"
    )

    print("=" * 80)


# ==============================================================================
# STEP 10 - SAVE RESULT
# ==============================================================================

saved_rows = []


def save_result(
    text,
    analysis
):

    original = analysis["original"]

    row = {

        "input_text":
            text,

        "original_prediction":
            original["label"],

        "original_xlmr_probability":
            original["xlmr_probability"],

        "original_semantic_probability":
            original["semantic_probability"],

        "original_hybrid_probability":
            original["hybrid_probability"],

        "evasion_detected":
            analysis["evasion_detected"],

        "final_decision":
            analysis["final_decision"],

        "reason":
            analysis["reason"]
    }

    saved_rows.append(
        row
    )

    pd.DataFrame(
        saved_rows
    ).to_csv(
        CSV_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ==============================================================================
# STEP 11 - INTERACTIVE DEMO
# ==============================================================================

print("\n")
print("=" * 80)
print("EVASION-AWARE REAL-TIME DEMO")
print("=" * 80)

print(
    "Enter social-media text."
)

print(
    "Type 'exit' or 'quit' to stop."
)

print(
    f"Results will be saved to:\n{CSV_FILE}"
)

print("=" * 80)


while True:

    try:

        user_text = input(
            "\nInput > "
        )

    except (
        EOFError,
        KeyboardInterrupt
    ):

        print()
        break


    if user_text.strip().lower() in {
        "exit",
        "quit"
    }:

        break


    user_text = textnorm(
        user_text
    )

    if not user_text:

        continue


    try:

        result = evasion_analysis(
            user_text
        )

        display_result(
            user_text,
            result
        )

        save_result(
            user_text,
            result
        )

    except Exception as e:

        print(
            "\nERROR:",
            str(e)
        )


print("\nDemo finished.")
print(
    f"Results saved to: {CSV_FILE}"
)