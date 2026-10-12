
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "results" / "multilingual_cross_dataset_final"
MODEL_DIR = ROOT / "final_model"
STEP24_ROOT = ROOT / "step24_semantic_final"
SEM_FILE = STEP24_ROOT / "semantic_model" / "semantic_classifier.joblib"
CAL_FILE = STEP24_ROOT / "hybrid_calibration.json"
CFG_FILE = STEP24_ROOT / "step24_config.json"

MAX_LENGTH = 96
BATCH_SIZE = 8


def normalize(x) -> str:
    """Same as textnorm() in Step 24: collapse whitespace only."""
    return "" if x is None else " ".join(str(x).strip().split())


def detect_language(s: str) -> str:
    """Same as detect_language() in Step 24."""
    for c in s:
        n = ord(c)
        if 0x0C80 <= n <= 0x0CFF:
            return "Kannada"
        if 0x0D00 <= n <= 0x0D7F:
            return "Malayalam"
        if 0x0B80 <= n <= 0x0BFF:
            return "Tamil"
    return "English"


def artifact_status() -> list[tuple[str, Path, bool]]:
    items = [
        ("Step 20 XLM-R model folder", MODEL_DIR),
        ("Step 24 semantic classifier", SEM_FILE),
        ("Step 24 calibration (weights + threshold)", CAL_FILE),
        ("Step 24 config", CFG_FILE),
    ]
    return [(n, p, p.exists()) for n, p in items]


class Predictor:
    def __init__(self, model_dir=None):  # model_dir kept for compatibility; ignored
        self.mode = "none"
        self.notes: list[str] = []
        self.xlmr_weight, self.semantic_weight, self.threshold = 0.55, 0.45, 0.65
        self.tokenizer = self.model = self.clf = None
        self._load()

    def _load(self):
        missing = [f"{n}: {p}" for n, p, ok in artifact_status() if not ok]
        if missing:
            self.notes.append("Missing Step 24 artifacts:")
            self.notes.extend(missing)
            self.notes.append("Run  python src/step24_final.py --train  once if these were never created.")
            return
        try:
            import joblib
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            torch.set_num_threads(min(4, __import__("os").cpu_count() or 4))
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
            self.model.eval()
            self.clf = joblib.load(SEM_FILE)
            cal = json.loads(CAL_FILE.read_text(encoding="utf-8"))
            self.threshold = float(cal["threshold"])
            self.xlmr_weight = float(cal["xlmr_weight"])
            self.semantic_weight = float(cal["semantic_weight"])
            self.mode = "hybrid"
            self.notes.append("Loaded frozen Step 20 XLM-R + Step 24 semantic classifier.")
            self.notes.append(
                f"Calibration: XLM-R weight {self.xlmr_weight:.2f}, semantic weight {self.semantic_weight:.2f}, threshold {self.threshold:.2f}"
            )
        except Exception as e:  # noqa: BLE001
            self.mode = "none"
            self.notes.append(f"Load failed: {e}")

    # ---- exact Step 24 computations (xlmr_prob + embed)
    def _batch(self, texts: list[str]):
        torch = self.torch
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            p = torch.softmax(self.model(**enc).logits, dim=-1)[:, 1]
            h = self.model.roberta(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        mx = h.masked_fill(mask == 0, torch.finfo(h.dtype).min).max(1).values
        emb = torch.cat([mean, mx], 1).numpy().astype(np.float32)
        return p.numpy(), emb

    def components(self, texts: list[str]) -> dict:
        if self.mode != "hybrid":
            raise RuntimeError("Step 24 model not loaded. Open Settings to see which artifact is missing.")
        texts = [normalize(t) or "." for t in texts]
        px, ps = [], []
        for i in range(0, len(texts), BATCH_SIZE):
            p, emb = self._batch(texts[i : i + BATCH_SIZE])
            px.extend(p.tolist())
            ps.extend(self.clf.predict_proba(emb)[:, 1].tolist())
        px, ps = np.asarray(px), np.asarray(ps)
        return {"xlmr": px, "semantic": ps, "final": self.xlmr_weight * px + self.semantic_weight * ps}

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.components(texts)["final"]

    def predict(self, texts: list[str]):
        comp = self.components(texts)
        out = []
        for i, t in enumerate(texts):
            hp = float(comp["final"][i])
            out.append(
                {
                    "text": t,
                    "language": detect_language(normalize(t)),
                    "probability": hp,
                    "xlmr_prob": float(comp["xlmr"][i]),
                    "semantic_prob": float(comp["semantic"][i]),
                    "label": "Offensive" if hp >= self.threshold else "Non-Offensive",
                }
            )
        return out


def find_model_dirs():  # compatibility stub (not used any more)
    return []
