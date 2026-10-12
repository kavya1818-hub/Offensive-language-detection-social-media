"""Port of Step 32 (src/step32_evasion_aware.py): same variants, same risk rule, same frozen Step 24 model."""
import re

ZW = "​"


def _core(word):
    return re.sub(r"^[^\w]+|[^\w]+$", "", word, flags=re.UNICODE)


def _word_variants(text, name, build):
    out = []
    words = text.split()
    for i, word in enumerate(words):
        core = _core(word)
        if len(core) < 5:
            continue
        start = word.find(core)
        new_core = build(core)
        if new_core is None:
            continue
        nw = words.copy()
        nw[i] = word[:start] + new_core + word[start + len(core):]
        v = " ".join(nw)
        if v != text:
            out.append((name, v))
    return out


def generate_variants(text):
    """Exactly the four Step 32 transformations (no dictionary, no keywords)."""

    def split_mid(core, ch):
        mid = len(core) // 2
        if mid <= 1 or mid >= len(core) - 1:
            return None
        return core[:mid] + ch + core[mid:]

    def repeat(core):
        mid = len(core) // 2
        if mid <= 1 or mid >= len(core) - 1:
            return None
        return core[:mid] + core[mid] * 3 + core[mid + 1:]

    def zw(core):
        return core[: len(core) // 2] + ZW + core[len(core) // 2:]

    variants = []
    variants += _word_variants(text, "Internal character spacing", lambda c: split_mid(c, " "))
    variants += _word_variants(text, "Internal punctuation", lambda c: split_mid(c, "."))
    variants += _word_variants(text, "Repeated-character variation", repeat)
    variants += _word_variants(text, "Zero-width insertion", zw)

    seen, uniq = set(), []
    for n, v in variants:
        if v not in seen:
            seen.add(v)
            uniq.append((n, v))
    return uniq


def analyze(predictor, text):
    original = predictor.predict([text])[0]
    variants = generate_variants(text)
    rows = []
    if variants:
        preds = predictor.predict([v for _, v in variants])
        for (name, v), p in zip(variants, preds):
            rows.append(
                {
                    "Transformation": name,
                    "Variant": v.replace(ZW, "<ZWSP>"),
                    "Hybrid prob.": round(p["probability"], 4),
                    "Prediction": p["label"],
                    "Prob. change": round(p["probability"] - original["probability"], 4),
                }
            )

    # Step 32 risk rule: original Non-Offensive, a variant becomes Offensive with >= +0.20 probability
    evasion, reason = False, "Stable prediction"
    if rows and original["label"] == "Non-Offensive":
        off = [r for r in rows if r["Prediction"] == "Offensive"]
        if off:
            best = max(off, key=lambda r: r["Hybrid prob."])
            if best["Hybrid prob."] - original["probability"] >= 0.20:
                evasion = True
                reason = "A normalized variant changed the prediction to Offensive with a significant probability increase."

    return {
        "original": original,
        "rows": rows,
        "evasion_detected": evasion,
        "reason": reason,
        "final_decision": "POSSIBLE EVASION - REVIEW" if evasion else original["label"],
        "flip_ratio": (sum(r["Prediction"] != original["label"] for r in rows) / len(rows)) if rows else 0.0,
        "max_shift": max((abs(r["Prob. change"]) for r in rows), default=0.0),
    }
