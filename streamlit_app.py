"""appendicitis decision-support prototype (frontend).

Run locally:   streamlit run streamlit_app.py
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from backend import UNKNOWN, AppendicitisPredictor, pretty

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Appendix AI", page_icon="🩺", layout="wide")


@st.cache_resource
def load_predictor():
    return AppendicitisPredictor()


@st.cache_data
def load_samples():
    return pd.read_csv(ROOT / "data" / "sample_patients.csv")


predictor = load_predictor()
info = predictor.info
samples = load_samples()
metrics = info["test_metrics"]

# ------------------------------------------------------------------ form layout
SECTIONS = {
    "👤 Patient": ["Age", "Sex", "Height", "Weight", "BMI"],
    "🤒 Symptoms": ["Migratory_Pain", "Lower_Right_Abd_Pain", "Nausea", "Loss_of_Appetite",
                   "Coughing_Pain", "Dysuria", "Stool"],
    "🩺 Physical examination": ["Body_Temperature", "Ipsilateral_Rebound_Tenderness",
                               "Contralateral_Rebound_Tenderness", "Psoas_Sign", "Peritonitis"],
    "🧪 Laboratory": ["WBC_Count", "Neutrophil_Percentage", "Neutrophilia", "CRP", "RBC_Count",
                     "Hemoglobin", "RDW", "Thrombocyte_Count", "Ketones_in_Urine", "RBC_in_Urine",
                     "WBC_in_Urine"],
    "🔊 Ultrasound report": ["US_Performed", "Appendix_on_US", "Appendix_Diameter", "Free_Fluids"],
}
UNITS = {"Age": "years", "Height": "cm", "Weight": "kg", "BMI": "kg/m²", "Body_Temperature": "°C",
         "WBC_Count": "×10³/µL", "Neutrophil_Percentage": "%", "CRP": "mg/L", "RBC_Count": "×10⁶/µL",
         "Hemoglobin": "g/dL", "RDW": "%", "Thrombocyte_Count": "×10³/µL", "Appendix_Diameter": "mm"}
assert sorted(sum(SECTIONS.values(), [])) == sorted(predictor.features), "form does not match model features"


def key(f):
    return f"in_{f}"


def clear_form():
    for f in predictor.features:
        st.session_state[key(f)] = None if f in predictor.numeric else UNKNOWN
    st.session_state.pop("result", None)
    st.session_state["loaded_sample"] = None


def load_sample():
    choice = st.session_state.get("sample_choice")
    if not choice or choice == "—":
        return
    row = samples[samples["sample_id"] == choice.split(" ·")[0]].iloc[0]
    for f in predictor.features:
        v = row[f]
        if f in predictor.numeric:
            st.session_state[key(f)] = None if pd.isna(v) else float(v)
        else:
            st.session_state[key(f)] = UNKNOWN if pd.isna(v) else str(v)
    st.session_state["loaded_sample"] = row["sample_id"]
    st.session_state["true_label"] = row["true_diagnosis"]
    st.session_state.pop("result", None)


if key(predictor.features[0]) not in st.session_state:
    clear_form()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🩺 AppendiScope")
    st.caption("AI decision support for pediatric appendicitis — research prototype")
    st.markdown("### Try a sample patient")
    st.selectbox("Real, anonymised patients the model has never seen",
                 ["—"] + [f"{r.sample_id} · true: {r.true_diagnosis}" for r in samples.itertuples()],
                 key="sample_choice", on_change=load_sample)
    st.button("Clear form", on_click=clear_form, width="stretch")
    st.divider()
    st.markdown("### Model")
    st.markdown(f"**{info['model_name']}** · {len(predictor.features)} inputs")
    c1, c2 = st.columns(2)
    c1.metric("Test accuracy", f"{metrics['Accuracy']:.1%}")
    c2.metric("ROC-AUC", f"{metrics['Macro ROC-AUC (OvR)']:.2f}")
    c1.metric("Appendicitis sens.", f"{metrics['Appendicitis sensitivity']:.0%}")
    c2.metric("Complicated sens.", f"{metrics['Complicated sensitivity']:.0%}")
    st.divider()
    st.warning("⚠️ Research prototype — **not a medical device**. Never use it to make clinical decisions.")

tab_predict, tab_perf, tab_about = st.tabs(["🔍 Patient assessment", "📊 Model performance", "ℹ️ About"])

# ------------------------------------------------------------------ assessment tab
with tab_predict:
    st.header("Patient assessment")
    st.caption("Fill in what you know — every field is optional. Pick a sample patient in the sidebar to try it quickly.")
    if st.session_state.get("loaded_sample"):
        st.info(f"Loaded **{st.session_state['loaded_sample']}** (true diagnosis: "
                f"**{st.session_state['true_label']}**). Edit any value, then press **Predict**.")

    with st.form("patient_form"):
        for title, fields in SECTIONS.items():
            with st.expander(title, expanded=True):
                cols = st.columns(4)
                for i, f in enumerate(fields):
                    label = pretty(f) + (f" ({UNITS[f]})" if f in UNITS else "")
                    with cols[i % 4]:
                        if f in predictor.numeric:
                            rng = info["numeric_features"][f]
                            st.number_input(label, value=None, step=0.1, format="%.2f", key=key(f),
                                            help=f"Typical range in training data: {rng['min']:g}–{rng['max']:g}. "
                                                 f"Median: {rng['median']:g}. Leave empty if unknown.")
                        else:
                            st.selectbox(label, [UNKNOWN] + info["categorical_features"][f], key=key(f))
        submitted = st.form_submit_button("🔍 Predict", type="primary", width="stretch")

    if submitted:
        patient = {f: st.session_state[key(f)] for f in predictor.features}
        if patient.get("BMI") is None and patient.get("Height") and patient.get("Weight"):
            patient["BMI"] = patient["Weight"] / (patient["Height"] / 100) ** 2
        st.session_state["result"] = {
            "patient": patient,
            "prediction": predictor.predict(patient),
            "warnings": predictor.validate(patient),
            "explanation": predictor.explain(patient),
        }

    res = st.session_state.get("result")
    if res:
        pred = res["prediction"]
        st.divider()
        st.subheader("Result")
        for w in res["warnings"]:
            st.warning(w)

        c1, c2, c3 = st.columns(3)
        c1.metric("Diagnosis", pred["diagnosis"], f"P(appendicitis) = {pred['appendicitis_probability']:.0%}",
                  delta_color="off")
        c2.metric("Most likely class", pred["predicted_class"],
                  f"{pred['probabilities'][pred['predicted_class']]:.0%} confidence", delta_color="off")
        c3.metric("Complicated / perforation risk", pred["complicated_risk"],
                  f"P(complicated) = {pred['complicated_probability']:.0%}", delta_color="off")

        box = {"No appendicitis": st.success, "Uncomplicated": st.warning, "Complicated": st.error}
        msg = {
            "No appendicitis": "The model finds **no appendicitis** most likely.",
            "Uncomplicated": "The model finds **uncomplicated appendicitis** most likely (no signs of complicated disease).",
            "Complicated": "The model finds **complicated appendicitis** most likely — this is the group that "
                           "includes perforated appendicitis.",
        }
        box[pred["predicted_class"]](msg[pred["predicted_class"]])

        if st.session_state.get("loaded_sample") and st.session_state.get("true_label"):
            ok = st.session_state["true_label"] == pred["predicted_class"]
            (st.success if ok else st.error)(
                f"{'✅ Correct' if ok else '❌ Incorrect'} — true diagnosis for this sample patient: "
                f"**{st.session_state['true_label']}**")

        left, right = st.columns(2)
        with left:
            st.markdown("**Probability of each class**")
            prob_df = pd.DataFrame({"Class": list(pred["probabilities"]),
                                    "Probability": list(pred["probabilities"].values())})
            st.bar_chart(prob_df, x="Class", y="Probability", horizontal=True, height=220)
        with right:
            st.markdown(f"**Why? Top factors for “{pred['predicted_class']}”**")
            exp = res["explanation"]
            if len(exp):
                exp = exp.assign(Direction=np.where(exp["Effect"] > 0, "▲ supports", "▼ against"))
                st.dataframe(exp[["Feature", "Value", "Direction", "Effect"]].round({"Effect": 3}),
                             hide_index=True, width="stretch")
                st.caption("SHAP contributions from the CatBoost model. ‘missing’ can matter because "
                           "missing tests were informative in the training data.")

        report = [
            "AppendiScope — prediction report",
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
            "RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE",
            "",
            f"Diagnosis: {pred['diagnosis']} (P = {pred['appendicitis_probability']:.1%})",
            f"Most likely class: {pred['predicted_class']}",
            f"Complicated/perforation risk: {pred['complicated_risk']} (P = {pred['complicated_probability']:.1%})",
            "",
            "Class probabilities:",
            *[f"  {c}: {p:.1%}" for c, p in pred["probabilities"].items()],
            "",
            "Inputs:",
            *[f"  {pretty(f)}: {'—' if v is None or v == UNKNOWN else v}" for f, v in res["patient"].items()],
        ]
        st.download_button("⬇️ Download report", "\n".join(report), file_name="appendiscope_report.txt",
                           width="stretch")

# ------------------------------------------------------------------ performance tab
with tab_perf:
    st.header("Model performance")
    st.caption(f"Locked test set of 156 patients that were never used for training or model selection. "
               f"95% confidence interval for accuracy: {info['test_accuracy_95ci'][0]:.1%}–{info['test_accuracy_95ci'][1]:.1%}.")
    st.dataframe(pd.Series(metrics).map("{:.3f}".format).rename("Test set").to_frame(), width="stretch")
    figs = sorted((ROOT / "figures").glob("*.png"))
    captions = {
        "01": "Model selection by cross-validation (red = chosen model)",
        "02": "Confusion matrix on the 156 test patients",
        "03": "ROC and precision–recall curves",
        "04": "Precision, recall and F1 per class",
        "05": "Accuracy confidence interval and clinical sensitivity/specificity",
        "06": "Cross-validation vs test accuracy",
        "07": "Most important variables (permutation importance)",
        "08": "Does ultrasound add value? Five setups on the same 578 patients",
        "09": "Accuracy across 25 cross-validation folds",
    }
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, fig in zip(cols, figs[i:i + 2]):
            col.image(str(fig), caption=captions.get(fig.name[:2], fig.stem), width="stretch")

# ------------------------------------------------------------------ about tab
with tab_about:
    st.header("About this project")
    st.markdown(f"""
**Goal.** Help clinicians (in research settings) estimate whether a child with abdominal pain has
appendicitis and, if so, whether it is **complicated** (the group that includes perforation).

**Data.** Regensburg Pediatric Appendicitis Dataset — 780 children with a confirmed diagnosis.
The model uses {len(predictor.features)} routinely collected variables: demographics, symptoms, physical
examination, blood and urine tests, and the structured ultrasound report.

**Model.** {info['model_name']}, tuned with 5-fold cross-validation on 624 training patients and tested once
on 156 locked patients. Raw ultrasound images were also studied (ResNet18 features), but they did not
improve accuracy, so the app uses the structured ultrasound report instead.

**Important limitations**
- Single hospital, retrospective data; no external validation yet.
- "Complicated" is the dataset's severity label. It covers advanced disease such as perforation but is
  **not a direct perforation test**.
- Complicated cases are detected less reliably ({metrics['Complicated sensitivity']:.0%} sensitivity) than
  appendicitis itself ({metrics['Appendicitis sensitivity']:.0%}).

**{info['disclaimer']}**
""")
