"""Offensive Language Detection - Streamlit Dashboard.

Run from the project root:   streamlit run dashboard/app.py
Default login: admin / Admin@123  (change it in Settings)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import auth
import robustness
from predictor import PROJECT_ROOT, Predictor, artifact_status

st.set_page_config(page_title="Offensive Language Detection", page_icon="🛡️", layout="wide")
auth.init_db()

# ------------------------------------------------------------------ styling
st.markdown(
    """
<style>
.block-container {padding-top: 2rem;}
.metric-card {background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px 18px;}
.metric-card h3 {margin:0;font-size:.85rem;color:#9ca3af;font-weight:500;}
.metric-card p {margin:4px 0 0;font-size:1.7rem;font-weight:700;color:#f9fafb;}
.badge-off {background:#7f1d1d;color:#fecaca;padding:6px 14px;border-radius:999px;font-weight:700;}
.badge-ok  {background:#14532d;color:#bbf7d0;padding:6px 14px;border-radius:999px;font-weight:700;}
.badge-rev {background:#78350f;color:#fde68a;padding:6px 14px;border-radius:999px;font-weight:700;}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ project facts (from your report)
OVERALL = {"Accuracy": 79.73, "Macro-F1": 73.32, "Offensive Precision": 60.31, "Offensive Recall": 60.19, "Offensive F1": 60.25}
LANGWISE = pd.DataFrame(
    {
        "Language": ["English", "Tamil", "Malayalam", "Kannada", "Overall"],
        "Accuracy": [100.00, 77.16, 86.13, 77.70, 79.73],
        "Macro-F1": [100.00, 68.49, 57.88, 73.07, 73.32],
        "Offensive F1": [100.00, 51.95, 23.38, 61.90, 60.25],
    }
)
BASELINE = {"Accuracy": 83.82, "Precision": 83.97, "Recall": 83.82, "F1-score": 83.79}
SAMPLES = {
    "English": "You are an idiot.",
    "Tamil": "நீ ஒரு முட்டாள்",
    "Malayalam": "നീ ഒരു വിഡ്ഢി ആണ്",
    "Kannada": "ನೀನು ಒಬ್ಬ ಮೂರ್ಖ",
    "Non-offensive": "Have a great day, congratulations on the new job!",
}


def card(title, value):
    st.markdown(f'<div class="metric-card"><h3>{title}</h3><p>{value}</p></div>', unsafe_allow_html=True)


def badge(label, review=False):
    if review:
        return '<span class="badge-rev">⚠ REVIEW (unstable)</span>'
    cls = "badge-off" if label == "Offensive" else "badge-ok"
    return f'<span class="{cls}">{label}</span>'


@st.cache_resource(show_spinner="Loading model (first time only)...")
def get_predictor(model_dir):
    return Predictor(model_dir)


# ------------------------------------------------------------------ login
def login_page():
    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("## 🛡️ Offensive Language Detection")
        st.caption("Multilingual detection - English, Tamil, Malayalam, Kannada")
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            with st.form("login"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Login", use_container_width=True):
                    user = auth.verify_user(u, p)
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
            st.caption("First run default: admin / Admin@123")
        with tab2:
            with st.form("register"):
                nu = st.text_input("Choose username")
                em = st.text_input("Email")
                p1 = st.text_input("Password", type="password", key="p1")
                p2 = st.text_input("Confirm password", type="password", key="p2")
                if st.form_submit_button("Create account", use_container_width=True):
                    if p1 != p2:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = auth.create_user(nu, em, p1)
                        (st.success if ok else st.error)(msg)


if "user" not in st.session_state:
    login_page()
    st.stop()

user = st.session_state.user
is_admin = user["role"] == "admin"

# ------------------------------------------------------------------ sidebar
pages = ["Home", "Detect", "Multilingual Demo", "Screenshot Analysis", "Robustness Check", "Model Performance", "History & Analytics", "Settings"]
if is_admin:
    pages.append("Admin")

with st.sidebar:
    st.markdown("### 🛡️ Offense Detector")
    st.write(f"👤 **{user['username']}** ({user['role']})")
    page = st.radio("Navigation", pages, label_visibility="collapsed")
    st.divider()
    if st.button("Logout", use_container_width=True):
        del st.session_state["user"]
        st.rerun()

predictor = get_predictor(None)
with st.sidebar:
    mode_name = {"hybrid": "Hybrid (XLM-R + Semantic)", "none": "No model"}[predictor.mode]
    st.caption(f"Active model: **{mode_name}**")


def need_model():
    if predictor.mode == "none":
        st.error("No trained model could be loaded. Open **Settings** to see details / pick a model folder.")
        st.stop()


# ------------------------------------------------------------------ pages
def page_home():
    st.title("Offensive Language Detection in Social Media")
    st.write("Hybrid **XLM-RoBERTa + semantic Logistic Regression** detector with an inference-time evasion-aware robustness layer.")
    c = st.columns(5)
    for col, (k, v) in zip(c, OVERALL.items()):
        with col:
            card(k, f"{v:.2f}%")
    st.write("")
    a, b = st.columns(2)
    with a:
        fig = px.bar(LANGWISE[LANGWISE.Language != "Overall"].melt("Language", var_name="Metric", value_name="Score"),
                     x="Language", y="Score", color="Metric", barmode="group", title="Language-wise performance (%)")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        counts = pd.DataFrame({"Language": ["Tamil", "Malayalam", "Kannada"], "Samples": [41732, 12596, 5646]})
        st.plotly_chart(px.pie(counts, names="Language", values="Samples", hole=.5, title="DravidianCodeMix language distribution"),
                        use_container_width=True)
    st.info("Use **Detect** for a single sentence, **Screenshot Analysis** for comment screenshots (or CSV files), and **Robustness Check** to test obfuscation stability.")


def page_detect():
    st.title("Detect offensive language")
    need_model()
    choice = st.selectbox("Load a sample", ["(type your own)"] + list(SAMPLES))
    text = st.text_area("Enter social-media text", value="" if choice.startswith("(") else SAMPLES[choice], height=120)
    if st.button("Analyze", type="primary") and text.strip():
        r = predictor.predict([text])[0]
        auth.log_prediction(user["username"], text, r["language"], r["label"], r["probability"], "single")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Prediction**")
        c1.markdown(badge(r["label"]), unsafe_allow_html=True)
        c2.metric("Offensive probability", f"{r['probability']:.2%}")
        c3.metric("Detected script/language", r["language"])
        fig = go.Figure(go.Indicator(mode="gauge+number", value=r["probability"] * 100,
                                     gauge={"axis": {"range": [0, 100]}, "threshold": {"line": {"color": "red", "width": 3}, "value": predictor.threshold * 100}}))
        fig.update_layout(height=260, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Decision threshold: {predictor.threshold} (red line). Mode: {mode_name}.")
        with st.expander("Branch breakdown (for diagnosis)", expanded=True):
            b1, b2, b3 = st.columns(3)
            b1.metric("XLM-R probability", "n/a" if r["xlmr_prob"] is None else f"{r['xlmr_prob']:.2%}")
            b2.metric("Semantic LR probability", "n/a" if r["semantic_prob"] is None else f"{r['semantic_prob']:.2%}")
            b3.metric(f"Hybrid ({predictor.xlmr_weight:.2f}/{predictor.semantic_weight:.2f})", f"{r['probability']:.2%}")
            st.caption("Same computation as `python src/step24_final.py` real-time mode.")


def page_multilingual():
    st.title("Multilingual prediction")
    need_model()
    st.write("Edit the sentences below (one per language) and run them together.")
    rows = pd.DataFrame({"Language": list(SAMPLES), "Text": list(SAMPLES.values())})
    edited = st.data_editor(rows, use_container_width=True, num_rows="dynamic")
    if st.button("Run all", type="primary"):
        edited = edited[edited["Text"].astype(str).str.strip() != ""]
        res = predictor.predict(edited["Text"].astype(str).tolist())
        out = pd.DataFrame({"Given language": edited["Language"].values, "Text": [r["text"] for r in res],
                            "Detected script": [r["language"] for r in res], "Label": [r["label"] for r in res],
                            "Offensive prob.": [round(r["probability"], 4) for r in res]})
        for r, lang in zip(res, edited["Language"]):
            auth.log_prediction(user["username"], r["text"], r["language"], r["label"], r["probability"], "multilingual")
        st.dataframe(out, use_container_width=True)
        st.caption("These are example inputs; accuracy varies by language (see Model Performance).")


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


def page_robust():
    st.title("Evasion-aware robustness check (Step 32)")
    need_model()
    st.caption("The frozen Step 24 detector is run on automatically generated obfuscated variants (internal spacing, punctuation, "
               "repeated characters, zero-width). Flags instability only - it does not determine user intention.")
    text = st.text_input("Sentence", "You are an idiot.")
    if st.button("Run robustness check", type="primary") and text.strip():
        res = robustness.analyze(predictor, text)
        o = res["original"]
        auth.log_prediction(user["username"], text, o["language"], o["label"], o["probability"], "robustness")
        a, b, c, d = st.columns(4)
        a.markdown("**Original prediction**")
        a.markdown(badge(o["label"]), unsafe_allow_html=True)
        b.metric("Hybrid probability", f"{o['probability']:.2%}")
        c.metric("Variants changing label", f"{res['flip_ratio']:.0%}")
        d.metric("Max prob. change", f"{res['max_shift']:.2f}")
        st.markdown("**Final decision:** " + (badge("", review=True) if res["evasion_detected"] else badge(o["label"])), unsafe_allow_html=True)
        st.caption("Reason: " + res["reason"])
        if res["rows"]:
            df = pd.DataFrame(res["rows"])
            st.dataframe(df, use_container_width=True)
            fig = px.bar(df, x="Variant", y="Hybrid prob.", color="Transformation", title="Hybrid probability per variant")
            fig.add_hline(y=predictor.threshold, line_dash="dash", annotation_text="threshold")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No meaningful transformation generated (words shorter than 5 characters are skipped, as in Step 32).")


def page_perf():
    st.title("Model performance")
    st.subheader("Final independent test (5,886 samples)")
    c = st.columns(5)
    for col, (k, v) in zip(c, OVERALL.items()):
        with col:
            card(k, f"{v:.2f}%")
    st.write("")
    st.dataframe(LANGWISE, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(LANGWISE.melt("Language", var_name="Metric", value_name="Score"), x="Language", y="Score", color="Metric", barmode="group"),
                    use_container_width=True)
    st.subheader("Baseline (TF-IDF + Logistic Regression)")
    st.dataframe(pd.DataFrame([BASELINE]), hide_index=True, use_container_width=True)
    st.write("**Errors:** 1,193 of 5,886 (~20.27%)")

    st.subheader("Saved result files from your project")
    res_dir = PROJECT_ROOT / "results"
    files = sorted(list(res_dir.rglob("*.csv")) + list(res_dir.rglob("*.png")))[:300] if res_dir.exists() else []
    if not files:
        st.caption("No files found in results/.")
        return
    pick = st.selectbox("Open a result file", [f.relative_to(res_dir).as_posix() for f in files])
    f = res_dir / pick
    if f.suffix == ".png":
        st.image(str(f))
    else:
        try:
            st.dataframe(pd.read_csv(f).head(500), use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Cannot display: {e}")


def page_history():
    st.title("History & analytics")
    scope = None if (is_admin and st.toggle("Show all users", False)) else user["username"]
    rows = auth.get_predictions(scope)
    if not rows:
        st.info("No predictions yet. Run something in Detect.")
        return
    df = pd.DataFrame(rows)
    c = st.columns(3)
    c[0].metric("Total predictions", len(df))
    c[1].metric("Offensive", int((df.label == "Offensive").sum()))
    c[2].metric("Non-offensive", int((df.label == "Non-Offensive").sum()))
    a, b = st.columns(2)
    a.plotly_chart(px.pie(df, names="label", hole=.5, title="Label split"), use_container_width=True)
    b.plotly_chart(px.histogram(df, x="language", color="label", barmode="group", title="By language"), use_container_width=True)
    df["created_at"] = pd.to_datetime(df["created_at"])
    daily = df.groupby([df.created_at.dt.date, "label"]).size().reset_index(name="count")
    st.plotly_chart(px.line(daily, x="created_at", y="count", color="label", markers=True, title="Activity over time"), use_container_width=True)
    st.dataframe(df, use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button("Download history CSV", df.to_csv(index=False).encode("utf-8"), "history.csv", "text/csv")
    if d2.button("Clear my history" if scope else "Clear ALL history"):
        auth.clear_predictions(scope)
        st.rerun()


def page_settings():
    st.title("Settings")
    st.subheader("Model Components")
    display_names = {
    "Step 20 XLM-R model folder": "Multilingual XLM-R Model",
    "Step 24 semantic classifier": "Semantic Classifier",
    "Step 24 calibration (weights + threshold)": "Hybrid Calibration",
    "Step 24 config": "Model Configuration",
    }
    for name, path, ok in artifact_status():
        name = display_names.get(name, name)
    st.write("**Loader notes**")
    for n in predictor.notes:
        st.write("•", n)
    st.subheader("Change password")
    with st.form("pw"):
        n1 = st.text_input("New password", type="password")
        n2 = st.text_input("Confirm", type="password")
        if st.form_submit_button("Update"):
            if n1 != n2:
                st.error("Passwords do not match.")
            else:
                ok, msg = auth.change_password(user["username"], n1)
                (st.success if ok else st.error)(msg)


def page_admin():
    st.title("Admin - user management")
    st.dataframe(pd.DataFrame(auth.list_users()), use_container_width=True, hide_index=True)
    st.subheader("Add user")
    with st.form("adduser"):
        u = st.text_input("Username")
        e = st.text_input("Email")
        p = st.text_input("Password", type="password")
        r = st.selectbox("Role", ["user", "admin"])
        if st.form_submit_button("Create"):
            ok, msg = auth.create_user(u, e, p, r)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()
    st.subheader("Delete user")
    names = [x["username"] for x in auth.list_users() if x["username"] != auth.DEFAULT_ADMIN_USER]
    if names:
        d = st.selectbox("User", names)
        if st.button("Delete user"):
            auth.delete_user(d)
            st.rerun()


{
    "Home": page_home, "Detect": page_detect, "Multilingual Demo": page_multilingual, "Screenshot Analysis": page_batch,
    "Robustness Check": page_robust, "Model Performance": page_perf, "History & Analytics": page_history,
    "Settings": page_settings, "Admin": page_admin,
}[page]()
