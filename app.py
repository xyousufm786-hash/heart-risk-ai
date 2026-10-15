"""Streamlit prototype: heart-disease risk estimate with calibrated probability, uncertainty and explanation.

Run:  streamlit run app/app.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrisk.data import HEART_DESCRIPTIONS  # noqa: E402
from hrisk.predictor import HeartRiskModel  # noqa: E402

st.set_page_config(page_title="Heart Risk AI", page_icon="❤️", layout="wide")


@st.cache_resource
def load():
    return HeartRiskModel()


model = load()
meta = model.meta

PRESETS = {
    "Custom": None,
    "Example A - younger, asymptomatic-free": dict(age=44, sex=0, cp=2, trestbps=118, chol=210, fbs=0, restecg=0, thalach=172, exang=0, oldpeak=0.0, slope=2, ca=0, thal=2),
    "Example B - middle-aged, mixed signals": dict(age=56, sex=1, cp=1, trestbps=135, chol=250, fbs=0, restecg=1, thalach=140, exang=0, oldpeak=1.2, slope=1, ca=1, thal=2),
    "Example C - older, multiple risk markers": dict(age=63, sex=1, cp=0, trestbps=150, chol=268, fbs=1, restecg=1, thalach=118, exang=1, oldpeak=2.8, slope=0, ca=2, thal=3),
}

st.title("❤️ Heart Disease Risk Estimator")
st.caption("Educational prototype for INT4203 - not a medical device. Logistic-regression model trained on the Cleveland "
           "heart-disease dataset (n = 302).")

with st.sidebar:
    st.header("Patient data")
    preset = st.selectbox("Load an example", list(PRESETS))
    d = PRESETS[preset] or dict(age=54, sex=1, cp=0, trestbps=130, chol=245, fbs=0, restecg=1, thalach=150, exang=0, oldpeak=1.0, slope=1, ca=0, thal=2)
    k = preset  # re-key widgets when the preset changes
    row = {}
    row["age"] = st.slider("Age (years)", 25, 80, int(d["age"]), key=f"age{k}")
    row["sex"] = 1 if st.radio("Sex", ["Female", "Male"], index=int(d["sex"]), horizontal=True, key=f"sex{k}") == "Male" else 0
    cp_opts = {"Asymptomatic": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Typical angina": 3}
    row["cp"] = cp_opts[st.selectbox("Chest-pain type", list(cp_opts), index=int(d["cp"]), key=f"cp{k}")]
    row["trestbps"] = st.slider("Resting blood pressure (mm Hg)", 90, 200, int(d["trestbps"]), key=f"bp{k}")
    row["chol"] = st.slider("Cholesterol (mg/dl)", 120, 570, int(d["chol"]), key=f"ch{k}")
    row["fbs"] = int(st.checkbox("Fasting blood sugar > 120 mg/dl", bool(d["fbs"]), key=f"fbs{k}"))
    ecg = {"Normal": 0, "ST-T abnormality": 1, "LV hypertrophy": 2}
    row["restecg"] = ecg[st.selectbox("Resting ECG", list(ecg), index=int(d["restecg"]), key=f"ecg{k}")]
    row["thalach"] = st.slider("Max heart rate achieved (bpm)", 70, 205, int(d["thalach"]), key=f"hr{k}")
    row["exang"] = int(st.checkbox("Exercise-induced angina", bool(d["exang"]), key=f"ex{k}"))
    row["oldpeak"] = st.slider("ST depression (oldpeak, mm)", 0.0, 6.5, float(d["oldpeak"]), 0.1, key=f"op{k}")
    sl = {"Downsloping": 0, "Flat": 1, "Upsloping": 2}
    row["slope"] = sl[st.selectbox("ST slope", list(sl), index=int(d["slope"]), key=f"sl{k}")]
    row["ca"] = st.selectbox("Major vessels coloured (0-3)", [0, 1, 2, 3], index=int(d["ca"]), key=f"ca{k}")
    th = {"Fixed defect": 1, "Normal": 2, "Reversible defect": 3}
    row["thal"] = th[st.selectbox("Thallium stress test", list(th), index=int(d["thal"]) - 1, key=f"th{k}")]

pred = model.predict(row)
col1, col2 = st.columns([1, 1.4], gap="large")

with col1:
    st.subheader("Risk estimate")
    colour = {"Low": "#2e7d32", "Moderate": "#f9a825", "High": "#c62828"}[pred.risk_band]
    st.markdown(
        f"<div style='padding:18px;border-radius:12px;background:{colour}15;border:2px solid {colour}'>"
        f"<div style='font-size:15px;color:#555'>Estimated probability of heart disease</div>"
        f"<div style='font-size:52px;font-weight:700;color:{colour}'>{pred.probability:.0%}</div>"
        f"<div style='font-size:20px;color:{colour}'>{pred.risk_band} risk</div></div>", unsafe_allow_html=True)
    st.progress(pred.probability)

    st.subheader("How sure is the model?")
    lab = {0: "no disease", 1: "disease"}
    for alpha, s in sorted(pred.conformal_set.items()):
        conf = int(round((1 - alpha) * 100))
        if len(s) == 1:
            msg = f"✅ **{conf}% confidence set: {lab[s[0]]}** - the model is confident at this level."
        elif len(s) == 2:
            msg = f"⚠️ **{conf}% confidence set: {{disease, no disease}}** - *abstain*: the evidence is ambiguous, refer to a clinician."
        else:
            msg = f"⚠️ **{conf}% confidence set: empty** - neither class reaches {conf}% confidence, so the model *abstains*."
        st.markdown(msg)
    if pred.out_of_range:
        st.warning("Input outside the training range for: " + ", ".join(pred.out_of_range) + ". Treat the estimate with extra caution.")

with col2:
    st.subheader("Why this estimate?")
    st.caption("Each bar is the exact contribution of a feature to the log-odds of disease (positive = pushes risk up).")
    c = pred.contributions.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.barh(c.index, c.values, color=["#c62828" if v > 0 else "#1565c0" for v in c.values])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Contribution to log-odds")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, clear_figure=True)
    with st.expander("Feature descriptions"):
        for f, t in HEART_DESCRIPTIONS.items():
            st.markdown(f"- **{f}**: {t}")

with st.expander("Model card"):
    st.markdown(
        f"- **Model:** {meta['model']}  \n- **Training data:** {meta['n']} patients (Cleveland), prevalence {meta['prevalence']:.1%}  \n"
        f"- **Nested-CV performance (25 outer folds):** ROC-AUC {meta['cv_auc']}, accuracy {meta['cv_acc']}  \n"
        f"- **Known limits:** single hospital, 1980s data, 302 patients, subgroup sizes are small; not validated prospectively.  \n"
        f"- **Intended use:** teaching / research demonstration only.")
