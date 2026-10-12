# ==============================================================================================
# SCREENSHOT ANALYSIS PAGE  -  paste into dashboard/app.py, REPLACING the whole old
# `def page_batch():` function (everything from `def page_batch():` down to just above
# `def page_robust():`).  Needs dashboard/ocr_utils.py next to app.py.
# ==============================================================================================


@st.cache_resource(show_spinner="Loading OCR model (first time it downloads ~100 MB, internet needed)...")
def get_easyocr(langs):
    import easyocr

    return easyocr.Reader(list(langs), gpu=False)


def _color_pred(col):
    return ["background-color:#7f1d1d;color:#fecaca" if v == "Offensive" else "background-color:#14532d;color:#bbf7d0" for v in col]


def _tesseract_help():
    with st.expander("How to install Tesseract on Windows", expanded=True):
        st.markdown(
            "1. Download the installer from **github.com/UB-Mannheim/tesseract/wiki** and run it.\n"
            "2. On the *Additional language data* step tick **Tamil, Malayalam, Kannada** (English is included).\n"
            "3. Run `pip install pytesseract` in your virtual environment.\n"
            "4. Restart Streamlit. If it is still not found, set the path once:  \n"
            "   `setx TESSERACT_CMD \"C:\\Program Files\\Tesseract-OCR\\tesseract.exe\"` and open a new terminal."
        )


def _screenshot_tab():
    import uuid

    import ocr_utils

    st.write(
        "Upload **screenshots of social-media comments**. The text is read with OCR, you can correct it, "
        "then every comment is checked by the Step 24 detector and marked on the image."
    )
    c1, c2 = st.columns(2)
    engine = c1.radio(
        "OCR engine", ["EasyOCR", "Tesseract"], horizontal=True,
        help="EasyOCR: pip only, English / Tamil / Kannada. Tesseract: separate install, also reads Malayalam.",
    )
    if engine == "EasyOCR":
        lang_opts = {"English": ["en"], "Tamil + English": ["ta", "en"], "Kannada + English": ["kn", "en"]}
    else:
        lang_opts = {"English": "eng", "Tamil + English": "tam+eng", "Malayalam + English": "mal+eng",
                     "Kannada + English": "kan+eng", "All four languages": "eng+tam+mal+kan"}
    lang = c2.selectbox("Language in the screenshot", list(lang_opts))
    c3, c4 = st.columns(2)
    sep = c3.slider("Comment separation", 0.0, 2.0, 0.8, 0.1,
                    help="Higher = merge more lines into one comment. 0 = every line is its own comment.")
    skip_first = c4.checkbox("Ignore first line of each comment (user name)", value=False)

    files = st.file_uploader("Screenshots (PNG / JPG / WEBP)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    if not files:
        st.caption("Tip: crop to just the comments, use a sharp screenshot, and pick the matching language.")
        return

    if st.button("1️⃣ Extract comments", type="primary"):
        imgs, all_lines = [], []
        try:
            with st.spinner("Reading text from the screenshot(s)..."):
                reader = get_easyocr(tuple(lang_opts[lang])) if engine == "EasyOCR" else None
                for f in files:
                    disp, oimg = ocr_utils.prepare_image(f.getvalue())
                    boxes = (ocr_utils.ocr_easyocr(reader, oimg) if engine == "EasyOCR"
                             else ocr_utils.ocr_tesseract(oimg, lang_opts[lang]))
                    imgs.append(disp)
                    all_lines.append(ocr_utils.boxes_to_lines(boxes))
        except ImportError as e:
            st.error(f"Missing package: {e}.  Run:  pip install easyocr pytesseract pillow")
            return
        except Exception as e:  # noqa: BLE001
            st.error(f"OCR failed: {e}")
            if engine == "Tesseract":
                _tesseract_help()
            return
        st.session_state.ocr = {"imgs": imgs, "lines": all_lines, "eid": uuid.uuid4().hex}
        st.session_state.pop("ocr_result", None)

    ocr = st.session_state.get("ocr")
    if not ocr:
        return

    rows, boxes = [], []
    for k, lines in enumerate(ocr["lines"]):
        for c in ocr_utils.group_comments(lines, sep, skip_first):
            rows.append({"Include": True, "Image": k + 1, "Comment": c["text"]})
            boxes.append(c["box"])
    if not rows:
        st.warning("No comment text found. Try the other OCR engine, the right language, or a larger / clearer screenshot.")
        return

    st.subheader("Extracted comments")
    st.caption("Fix any OCR mistakes and untick rows that are not comments (names, buttons, captions).")
    edited = st.data_editor(
        pd.DataFrame(rows), key=f"ocr_editor_{ocr['eid']}_{sep}_{skip_first}", hide_index=True, num_rows="fixed",
        use_container_width=True, disabled=["Image"],
        column_config={"Include": st.column_config.CheckboxColumn(width="small"),
                       "Image": st.column_config.NumberColumn(width="small"),
                       "Comment": st.column_config.TextColumn(width="large")},
    )

    if st.button("2️⃣ Detect offensive language", type="primary"):
        sel = edited[edited["Include"] & (edited["Comment"].astype(str).str.strip() != "")]
        if sel.empty:
            st.warning("Nothing selected.")
            return
        texts = sel["Comment"].astype(str).tolist()
        preds, bar = [], st.progress(0.0)
        for i in range(0, len(texts), 8):
            preds.extend(predictor.predict(texts[i : i + 8]))
            bar.progress(min(1.0, (i + 8) / len(texts)))
        bar.empty()
        res = pd.DataFrame({
            "Image": sel["Image"].values,
            "Comment": texts,
            "Language": [p["language"] for p in preds],
            "Prediction": [p["label"] for p in preds],
            "Offensive %": [round(p["probability"] * 100, 2) for p in preds],
            "XLM-R %": [round(p["xlmr_prob"] * 100, 2) for p in preds],
            "Semantic %": [round(p["semantic_prob"] * 100, 2) for p in preds],
        })
        for p in preds:
            auth.log_prediction(user["username"], p["text"], p["language"], p["label"], p["probability"], "screenshot")
        st.session_state.ocr_result = {"res": res, "boxes": [boxes[i] for i in sel.index], "eid": ocr["eid"]}

    out = st.session_state.get("ocr_result")
    if not out or out["eid"] != ocr["eid"]:
        return
    res = out["res"]
    n_off = int((res["Prediction"] == "Offensive").sum())
    st.subheader("Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Comments checked", len(res))
    m2.metric("Offensive", n_off)
    m3.metric("Non-offensive", len(res) - n_off)
    for k, img in enumerate(ocr["imgs"]):
        items = [(out["boxes"][i], res["Prediction"].iloc[i], res["Offensive %"].iloc[i] / 100)
                 for i in range(len(res)) if res["Image"].iloc[i] == k + 1]
        st.image(ocr_utils.annotate(img, items), caption=f"Image {k + 1}  (red = offensive, green = non-offensive)")
    st.dataframe(res.style.apply(_color_pred, subset=["Prediction"]), use_container_width=True, hide_index=True)
    a, b = st.columns(2)
    a.plotly_chart(px.pie(res, names="Prediction", hole=.5, color="Prediction", title="Prediction split",
                          color_discrete_map={"Offensive": "#dc2626", "Non-Offensive": "#16a34a"}), use_container_width=True)
    b.plotly_chart(px.histogram(res, x="Offensive %", nbins=20, title="Offensive probability"), use_container_width=True)
    st.download_button("Download results CSV", res.to_csv(index=False).encode("utf-8-sig"), "screenshot_predictions.csv", "text/csv")


def _csv_tab():
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        st.caption("The CSV needs a text column. An optional `label` column (0/1 or Offensive/Non-Offensive) enables accuracy metrics.")
        return
    df = pd.read_csv(up)
    col = st.selectbox("Text column", df.columns, index=list(df.columns).index("text") if "text" in df.columns else 0)
    limit = st.slider("Max rows to process (CPU is slow)", 10, min(5000, max(len(df), 10)), min(200, len(df)))
    if st.button("Run batch", type="primary"):
        sub = df.head(limit).copy()
        bar = st.progress(0.0)
        probs = []
        texts = sub[col].astype(str).tolist()
        for i in range(0, len(texts), 32):
            probs.extend(predictor.predict_proba(texts[i : i + 32]).tolist())
            bar.progress(min(1.0, (i + 32) / len(texts)))
        sub["offensive_prob"] = probs
        sub["prediction"] = ["Offensive" if p >= predictor.threshold else "Non-Offensive" for p in probs]
        sub["language"] = [__import__("predictor").detect_language(t) for t in texts]
        for t, l, p, pr in zip(texts[:200], sub["language"][:200], sub["prediction"][:200], probs[:200]):
            auth.log_prediction(user["username"], t, l, p, pr, "batch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(sub, names="prediction", title="Prediction split", hole=.5), use_container_width=True)
        c2.plotly_chart(px.histogram(sub, x="offensive_prob", nbins=30, title="Probability distribution"), use_container_width=True)
        if "label" in sub.columns:
            from sklearn.metrics import accuracy_score, classification_report, f1_score

            y = sub["label"].map(lambda v: 1 if str(v).strip().lower() in ("1", "offensive", "true") else 0)
            yp = (sub["prediction"] == "Offensive").astype(int)
            st.metric("Accuracy", f"{accuracy_score(y, yp):.2%}")
            st.metric("Macro-F1", f"{f1_score(y, yp, average='macro'):.2%}")
            st.text(classification_report(y, yp, zero_division=0))
        st.dataframe(sub, use_container_width=True)
        st.download_button("Download results CSV", sub.to_csv(index=False).encode("utf-8"), "batch_predictions.csv", "text/csv")


def page_batch():
    st.title("Screenshot analysis")
    need_model()
    tab_img, tab_csv = st.tabs(["📷 Screenshot (OCR)", "📄 CSV file"])
    with tab_img:
        _screenshot_tab()
    with tab_csv:
        _csv_tab()
