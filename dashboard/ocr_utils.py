"""OCR helpers: social-media screenshot  ->  list of comments (text + position).

Two engines:
  * EasyOCR   (pip only)           English, Tamil, Kannada   (EasyOCR has no Malayalam)
  * Tesseract (separate install)   English, Tamil, Malayalam, Kannada

No offensive-word list is used anywhere. The only filtering is of generic UI text
(Like / Reply / timestamps / counts) that is not part of a comment.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

MIN_W, MAX_W = 1000, 2400

# --------------------------------------------------------------------------- image
def prepare_image(data: bytes):
    """Return (display_image, ocr_image). Small screenshots are upscaled; dark-mode ones are inverted for OCR."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    w, h = img.size
    scale = 1.0
    if w < MIN_W:
        scale = min(MIN_W / w, 3.0)
    elif w > MAX_W:
        scale = MAX_W / w
    if scale != 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    ocr_img = img
    if np.asarray(img.convert("L")).mean() < 100:      # dark mode: light text on dark background
        ocr_img = ImageOps.invert(img)
    return img, ocr_img


# --------------------------------------------------------------------------- engines
def find_tesseract() -> str | None:
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if cmd:
        return cmd
    for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if Path(p).exists():
            return p
    return None


def ocr_easyocr(reader, img: Image.Image):
    """-> list of (x0, y0, x1, y1, text) phrase boxes."""
    out = []
    for pts, text, conf in reader.readtext(np.array(img), paragraph=False):
        text = str(text).strip()
        if not text or conf < 0.2:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out.append((min(xs), min(ys), max(xs), max(ys), text))
    return out


def ocr_tesseract(img: Image.Image, lang: str):
    """-> list of (x0, y0, x1, y1, text) word boxes."""
    import pytesseract

    cmd = find_tesseract()
    if not cmd:
        raise RuntimeError("Tesseract is not installed (or not found).")
    pytesseract.pytesseract.tesseract_cmd = cmd
    d = pytesseract.image_to_data(img, lang=lang, config="--psm 6", output_type=pytesseract.Output.DICT)
    out = []
    for i, t in enumerate(d["text"]):
        t = str(t).strip()
        try:
            conf = float(d["conf"][i])
        except ValueError:
            conf = -1
        if t and conf >= 20:
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            out.append((x, y, x + w, y + h, t))
    return out


# --------------------------------------------------------------------------- boxes -> lines -> comments
def boxes_to_lines(boxes):
    """Merge word/phrase boxes that sit on the same row into text lines."""
    rows = []
    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        cy, h = (b[1] + b[3]) / 2, b[3] - b[1]
        for r in rows:
            if abs(r["cy"] - cy) < 0.5 * max(r["h"], h):
                r["items"].append(b)
                r["cy"] = float(np.mean([(i[1] + i[3]) / 2 for i in r["items"]]))
                r["h"] = float(np.median([i[3] - i[1] for i in r["items"]]))
                break
        else:
            rows.append({"cy": cy, "h": h, "items": [b]})
    lines = []
    for r in rows:
        items = sorted(r["items"], key=lambda b: b[0])
        lines.append((min(i[0] for i in items), min(i[1] for i in items), max(i[2] for i in items), max(i[3] for i in items),
                      " ".join(i[4] for i in items)))
    return sorted(lines, key=lambda l: l[1])


_TOKENS = {
    "like", "likes", "reply", "replies", "share", "send", "follow", "following", "report", "edited", "translate",
    "translation", "see", "more", "view", "hide", "pinned", "top", "fan", "author", "most", "relevant", "newest",
    "ago", "comment", "comments", "repl", "·", "•", "|", "-", "–", "...", "…", "just", "now", "yesterday",
}
_TIME = re.compile(r"^\d+(?:[.,]\d+)?(?:s|m|h|d|w|y|k|mo|min|mins|hr|hrs|sec|secs|hour|hours|day|days|week|weeks|month|months|year|years)?$", re.I)


def is_noise(text: str) -> bool:
    """Generic social-media interface text (not a comment): Like / Reply / '2h' / '1.2K' ..."""
    # count letters/digits/combining marks (Indic vowel signs are marks, which \w would miss)
    if sum(1 for c in text if unicodedata.category(c)[0] in "LNM") < 2:
        return True
    toks = [t for t in re.split(r"[\s·•|]+", text.strip().lower()) if t]
    return not toks or all(t in _TOKENS or _TIME.match(t) for t in toks)


def group_comments(lines, separation: float = 0.8, skip_first_line: bool = False):
    """Group consecutive lines into comments.

    separation: a new comment starts when the vertical gap is larger than  separation x line height
                (or the left edge shifts a lot, e.g. a reply). 0 = every line is its own comment.
    skip_first_line: ignore the first line of multi-line blocks (usually the user name).
    """
    lines = [l for l in lines if not is_noise(l[4])]
    blocks = []
    for ln in lines:
        lh = ln[3] - ln[1]
        if blocks and separation > 0:
            last = blocks[-1][-1]
            ref = max(lh, last[3] - last[1])
            gap = ln[1] - last[3]
            if gap <= separation * ref and abs(ln[0] - blocks[-1][0][0]) <= 2.0 * ref:
                blocks[-1].append(ln)
                continue
        blocks.append([ln])
    comments = []
    for b in blocks:
        use = b[1:] if (skip_first_line and len(b) >= 2) else b
        text = " ".join(l[4] for l in use).strip()
        if not text:
            continue
        comments.append(
            {
                "text": text,
                "box": (min(l[0] for l in b), min(l[1] for l in b), max(l[2] for l in b), max(l[3] for l in b)),
            }
        )
    return comments


# --------------------------------------------------------------------------- drawing
def annotate(img: Image.Image, items):
    """items: list of (box, label, probability). Offensive = red, Non-Offensive = green."""
    out = img.copy()
    dr = ImageDraw.Draw(out)
    lw = max(2, out.width // 400)
    try:
        font = ImageFont.load_default(size=max(14, out.width // 70))
    except TypeError:  # old Pillow
        font = ImageFont.load_default()
    for box, label, prob in items:
        col = (220, 38, 38) if label == "Offensive" else (22, 163, 74)
        x0, y0, x1, y1 = [int(v) for v in box]
        dr.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline=col, width=lw)
        tag = f"{label} {prob:.0%}"
        tb = dr.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y0 - th - 10)
        dr.rectangle([x0 - 4, ty, x0 + tw + 8, ty + th + 6], fill=col)
        dr.text((x0, ty + 1), tag, fill=(255, 255, 255), font=font)
    return out
