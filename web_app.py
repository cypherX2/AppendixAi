"""AppendixAi — HTML/CSS/JavaScript website served by Flask.

Run locally (works offline):   python web_app.py   then open http://localhost:5000

The server only does three things:
  1. serves the web page (web/index.html, web/style.css, web/app.js)
  2. answers /api/... requests from the page with JSON
  3. serves the evaluation graphs from figures/
The model runs through backend/lite.py (NumPy only), so this website needs just Flask + NumPy
and fits easily on Vercel. Predictions are identical to the full CatBoost model.
"""
import csv
import math
from pathlib import Path
from urllib.parse import unquote

from flask import Flask, abort, jsonify, request, send_from_directory

from backend import LitePredictor, pretty

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
FIG_DIR = ROOT / "figures"

app = Flask(__name__, static_folder=None)
app.json.sort_keys = False          # keep class order: No appendicitis → Uncomplicated → Complicated
predictor = LitePredictor()
with open(ROOT / "data" / "sample_patients.csv", newline="", encoding="utf-8") as fh:
    samples = list(csv.DictReader(fh))

SECTIONS = [
    {"title": "👤 Patient", "fields": ["Age", "Sex", "Height", "Weight", "BMI"]},
    {"title": "🤒 Symptoms", "fields": ["Migratory_Pain", "Lower_Right_Abd_Pain", "Nausea", "Loss_of_Appetite",
                                       "Coughing_Pain", "Dysuria", "Stool"]},
    {"title": "🩺 Physical examination", "fields": ["Body_Temperature", "Ipsilateral_Rebound_Tenderness",
                                                   "Contralateral_Rebound_Tenderness", "Psoas_Sign", "Peritonitis"]},
    {"title": "🧪 Laboratory", "fields": ["WBC_Count", "Neutrophil_Percentage", "Neutrophilia", "CRP", "RBC_Count",
                                         "Hemoglobin", "RDW", "Thrombocyte_Count", "Ketones_in_Urine",
                                         "RBC_in_Urine", "WBC_in_Urine"]},
    {"title": "🔊 Ultrasound report", "fields": ["US_Performed", "Appendix_on_US", "Appendix_Diameter",
                                                "Free_Fluids"]},
]
UNITS = {"Age": "years", "Height": "cm", "Weight": "kg", "BMI": "kg/m²", "Body_Temperature": "°C",
         "WBC_Count": "×10³/µL", "Neutrophil_Percentage": "%", "CRP": "mg/L", "RBC_Count": "×10⁶/µL",
         "Hemoglobin": "g/dL", "RDW": "%", "Thrombocyte_Count": "×10³/µL", "Appendix_Diameter": "mm"}
FIGURE_CAPTIONS = {
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
assert sorted(f for s in SECTIONS for f in s["fields"]) == sorted(predictor.features), "form does not match model"


def clean(value):
    """Turn empty/NaN values into None so they become JSON null."""
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def sample_value(field, text):
    """CSV cells are text: convert numeric fields to numbers, empty cells to None."""
    if text is None or text.strip() == "":
        return None
    return float(text) if field in predictor.numeric else text


# ------------------------------------------------------------------ pages and files
@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(WEB_DIR, name)


@app.get("/figures/<path:name>")
def figures(name):
    return send_from_directory(FIG_DIR, name)


# ------------------------------------------------------------------ API
@app.get("/api/config")
def config():
    info = predictor.info
    fields = {}
    for f in predictor.features:
        entry = {"label": pretty(f) + (f" ({UNITS[f]})" if f in UNITS else "")}
        if f in predictor.numeric:
            rng = info["numeric_features"][f]
            entry.update(type="number", min=rng["min"], max=rng["max"], median=rng["median"])
        else:
            entry.update(type="select", options=info["categorical_features"][f])
        fields[f] = entry
    figs = sorted(p.name for p in FIG_DIR.glob("*.png"))
    return jsonify({
        "model_name": info["model_name"],
        "n_features": len(predictor.features),
        "classes": predictor.classes,
        "metrics": info["test_metrics"],
        "accuracy_ci": info["test_accuracy_95ci"],
        "disclaimer": info["disclaimer"],
        "sections": SECTIONS,
        "fields": fields,
        "figures": [{"file": f, "caption": FIGURE_CAPTIONS.get(f[:2], f)} for f in figs],
        "samples": [{"id": r["sample_id"], "true_diagnosis": r["true_diagnosis"]} for r in samples],
    })


@app.get("/api/sample")                    # /api/sample?id=Patient%201  (used by the page)
@app.get("/api/samples/<path:sample_id>")  # older style, kept for compatibility
def sample(sample_id=None):
    # Some hosts (e.g. Vercel) pass the path still URL-encoded ("Patient%201"), so decode it ourselves
    sample_id = unquote(request.args.get("id") or sample_id or "").replace("+", " ").strip()
    row = next((r for r in samples if r["sample_id"] == sample_id), None)
    if row is None:
        return jsonify({"error": f"Sample patient '{sample_id}' not found"}), 404
    return jsonify({"id": sample_id, "true_diagnosis": row["true_diagnosis"],
                    "patient": {f: sample_value(f, row.get(f)) for f in predictor.features}})


@app.post("/api/predict")
def predict():
    patient = (request.get_json(silent=True) or {}).get("patient", {})
    patient = {f: patient.get(f) for f in predictor.features}
    if patient.get("BMI") in (None, "") and patient.get("Height") and patient.get("Weight"):
        patient["BMI"] = float(patient["Weight"]) / (float(patient["Height"]) / 100) ** 2
    try:
        result = predictor.predict(patient)
        warnings = predictor.validate(patient)
        explanation = predictor.explain(patient)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Could not read the input: {exc}"}), 400
    return jsonify({
        "prediction": result,
        "warnings": warnings,
        "explanation": explanation,
        "patient": {f: clean(v) for f, v in patient.items()},
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"AppendixAi is running - open http://localhost:{port}  (press Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=port, debug=False)
