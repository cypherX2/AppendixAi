"""Backend: loads the trained model and turns patient data into predictions + explanations.

The web page (streamlit_app.py) only collects inputs and shows results.
Everything medical/ML lives here, so the same backend could later serve a REST API or mobile app.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
UNKNOWN = "Unknown"


def pretty(name: str) -> str:
    return name.replace("_", " ").replace("Abd", "Abdominal").replace("US", "Ultrasound")


class AppendicitisPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model = joblib.load(model_dir / "appendicitis_model.joblib")
        self.info = json.loads((model_dir / "model_info.json").read_text(encoding="utf-8"))
        self.features = self.info["features"]
        self.numeric = list(self.info["numeric_features"])
        self.categorical = list(self.info["categorical_features"])
        self.classes = self.info["classes"]

    # ------------------------------------------------------------------ input handling
    def to_frame(self, patient: dict) -> pd.DataFrame:
        row = {}
        for f in self.features:
            v = patient.get(f)
            if v is None or v == "" or v == UNKNOWN or (isinstance(v, float) and np.isnan(v)):
                row[f] = np.nan
            elif f in self.numeric:
                row[f] = float(v)
            else:
                row[f] = str(v)
        frame = pd.DataFrame([row], columns=self.features)
        frame[self.categorical] = frame[self.categorical].astype(object)
        return frame

    def validate(self, patient: dict) -> list:
        """Return human-readable warnings (the model still runs; missing values are allowed)."""
        warnings = []
        frame = self.to_frame(patient)
        filled = int(frame.notna().sum(axis=1).iloc[0])
        if filled < 8:
            warnings.append(f"Only {filled} of {len(self.features)} fields are filled in — the prediction will be unreliable.")
        for f, rng in self.info["numeric_features"].items():
            v = frame[f].iloc[0]
            if pd.notna(v) and not (rng["min"] <= v <= rng["max"]):
                warnings.append(f"{pretty(f)} = {v:g} is outside the training range "
                                f"({rng['min']:g}–{rng['max']:g}).")
        for key in ["CRP", "WBC_Count", "Appendix_Diameter"]:
            if pd.isna(frame[key].iloc[0]):
                warnings.append(f"{pretty(key)} is missing — it is one of the most important inputs.")
        return warnings

    # ------------------------------------------------------------------ prediction
    def predict(self, patient: dict) -> dict:
        probs = self.model.predict_proba(self.to_frame(patient))[0]
        idx = int(np.argmax(probs))
        p_app = float(probs[1] + probs[2])
        p_comp = float(probs[2])
        risk = "High" if p_comp >= 0.5 else "Moderate" if p_comp >= 0.2 else "Low"
        return {
            "predicted_class": self.classes[idx],
            "predicted_index": idx,
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "appendicitis_probability": p_app,
            "diagnosis": "Appendicitis likely" if p_app >= 0.5 else "Appendicitis unlikely",
            "complicated_probability": p_comp,
            "complicated_risk": risk,
            "severity": (None if idx == 0 else "Complicated" if idx == 2 else "Uncomplicated"),
        }

    # ------------------------------------------------------------------ explanation
    def explain(self, patient: dict, top: int = 8) -> pd.DataFrame:
        """Per-patient SHAP contributions (CatBoost), summed back to the original input fields."""
        pipe = self.model
        steps = getattr(pipe, "named_steps", {})
        prep, clf = steps.get("prep"), steps.get("clf")
        if prep is None or not hasattr(clf, "get_feature_importance"):
            return pd.DataFrame(columns=["Feature", "Value", "Effect"])
        from catboost import Pool

        frame = self.to_frame(patient)
        Xt = prep.transform(frame)
        names = prep.get_feature_names_out()
        cls = int(np.argmax(pipe.predict_proba(frame)[0]))
        shap = clf.get_feature_importance(Pool(Xt), type="ShapValues")[0, cls, :-1]

        totals = {}
        for name, value in zip(names, shap):
            block, raw = name.split("__", 1)
            if block == "num":
                raw = raw.replace("missingindicator_", "")
            else:
                raw = max((f for f in self.categorical if raw.startswith(f + "_")), key=len)
            totals[raw] = totals.get(raw, 0.0) + float(value)

        out = pd.DataFrame({"Feature": list(totals), "Effect": list(totals.values())})
        def show(v):
            if pd.isna(v):
                return "missing"
            return f"{v:.4g}" if isinstance(v, (int, float, np.floating)) else str(v)

        out["Value"] = [show(frame[f].iloc[0]) for f in out["Feature"]]
        out = out.reindex(out["Effect"].abs().sort_values(ascending=False).index).head(top)
        out["Feature"] = out["Feature"].map(pretty)
        return out[["Feature", "Value", "Effect"]].reset_index(drop=True)
