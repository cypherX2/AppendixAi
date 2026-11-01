"""Lightweight backend: runs the exported model (model/lite_model.json) with NumPy only.

Same interface and same results as backend/predictor.py, but without CatBoost, scikit-learn or
pandas, so the website stays far below Vercel's 500 MB size limit.
"""
import json
import math
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
UNKNOWN = "Unknown"


def pretty(name: str) -> str:
    return name.replace("_", " ").replace("Abd", "Abdominal").replace("US", "Ultrasound")


def _is_missing(v) -> bool:
    return v is None or v == "" or v == UNKNOWN or (isinstance(v, float) and math.isnan(v))


class LitePredictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        lite = json.loads((model_dir / "lite_model.json").read_text(encoding="utf-8"))
        self.info = json.loads((model_dir / "model_info.json").read_text(encoding="utf-8"))
        self.features = self.info["features"]
        self.numeric = list(self.info["numeric_features"])
        self.categorical = list(self.info["categorical_features"])
        self.classes = self.info["classes"]

        p = lite["preprocessing"]
        self.num_cols = p["numeric_columns"]
        self.medians = np.array(p["medians"])
        self.indicator_cols = p["indicator_columns"]
        self.mean = np.array(p["scaler_mean"])
        self.scale = np.array(p["scaler_scale"])
        self.cat_cols = p["categorical_columns"]
        self.categories = p["categories"]
        self.fill_value = p["fill_value"]
        self.output_names = p["output_names"]

        trees = lite["trees"]
        self.depth = len(trees[0]["features"])
        assert all(len(t["features"]) == self.depth for t in trees), "all trees must have the same depth"
        self.F = np.array([t["features"] for t in trees])                 # (T, depth) input column per split
        self.B = np.array([t["borders"] for t in trees], dtype=np.float32)  # (T, depth) thresholds (CatBoost uses float32)
        self.V = np.array([t["values"] for t in trees])                   # (T, leaves, classes)
        self.R = np.array([t["cover_ratio"] for t in trees])              # (T, leaves, depth)
        self.scale_out = lite["scale_and_bias"][0]
        self.bias = np.array(lite["scale_and_bias"][1])
        self.powers = 1 << np.arange(self.depth)
        leaves = np.arange(2 ** self.depth)
        self.leaf_bits = (leaves[:, None] >> np.arange(self.depth)) & 1   # (leaves, depth)

        # raw input field behind each of the 90 model inputs (used to sum explanations)
        self.output_to_field = []
        for name in self.output_names:
            block, raw = name.split("__", 1)
            if block == "num":
                raw = raw.replace("missingindicator_", "")
            else:
                raw = max((f for f in self.cat_cols if raw.startswith(f + "_")), key=len)
            self.output_to_field.append(raw)

    # ------------------------------------------------------------------ input handling
    def clean(self, patient: dict) -> dict:
        row = {}
        for f in self.features:
            v = patient.get(f)
            if _is_missing(v):
                row[f] = None
            elif f in self.numeric:
                row[f] = float(v)
            else:
                row[f] = str(v)
        return row

    def transform(self, row: dict) -> np.ndarray:
        raw = [row[c] for c in self.num_cols]
        imputed = [self.medians[i] if v is None else v for i, v in enumerate(raw)]
        flags = [1.0 if raw[i] is None else 0.0 for i in self.indicator_cols]
        num = (np.array(imputed + flags) - self.mean) / self.scale
        cat = []
        for col, cats in zip(self.cat_cols, self.categories):
            value = self.fill_value if row[col] is None else row[col]
            cat.extend(1.0 if value == c else 0.0 for c in cats)
        return np.concatenate([num, np.array(cat)])

    def validate(self, patient: dict) -> list:
        warnings = []
        row = self.clean(patient)
        filled = sum(v is not None for v in row.values())
        if filled < 8:
            warnings.append(f"Only {filled} of {len(self.features)} fields are filled in — the prediction will be unreliable.")
        for f, rng in self.info["numeric_features"].items():
            v = row[f]
            if v is not None and not (rng["min"] <= v <= rng["max"]):
                warnings.append(f"{pretty(f)} = {v:g} is outside the training range ({rng['min']:g}–{rng['max']:g}).")
        for key in ["CRP", "WBC_Count", "Appendix_Diameter"]:
            if row[key] is None:
                warnings.append(f"{pretty(key)} is missing — it is one of the most important inputs.")
        return warnings

    # ------------------------------------------------------------------ prediction
    def _leaves(self, x):
        bits = x.astype(np.float32)[self.F] > self.B                      # (T, depth), compared in float32 like CatBoost
        return bits.astype(int) @ self.powers, bits

    def predict_proba(self, patient: dict) -> np.ndarray:
        x = self.transform(self.clean(patient))
        leaves, _ = self._leaves(x)
        raw = self.scale_out * self.V[np.arange(len(leaves)), leaves].sum(axis=0) + self.bias
        e = np.exp(raw - raw.max())
        return e / e.sum()

    def predict(self, patient: dict) -> dict:
        probs = self.predict_proba(patient)
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

    # ------------------------------------------------------------------ explanation (exact TreeSHAP)
    def shap_values(self, x: np.ndarray) -> np.ndarray:
        """Path-dependent TreeSHAP for oblivious trees → (n_inputs, n_classes) on the raw score scale."""
        phi = np.zeros((len(x), self.V.shape[2]))
        _, bits = self._leaves(x)
        match = self.leaf_bits[None, :, :] == bits[:, None, :]           # (T, leaves, depth)
        for t in range(len(self.F)):
            feats = self.F[t]
            uniq = sorted(set(feats.tolist()))
            m = len(uniq)
            pos = np.array([uniq.index(f) for f in feats])                # which unique feature each split uses
            subsets = np.arange(2 ** m)
            in_s = ((subsets[:, None] >> pos[None, :]) & 1).astype(bool)   # (subsets, depth)
            w = np.where(in_s[:, None, :], match[t][None], self.R[t][None]).prod(axis=2)  # (subsets, leaves)
            v = w @ self.V[t]                                             # (subsets, classes)
            sizes = np.array([bin(s).count("1") for s in subsets])
            for u, feat in enumerate(uniq):
                without = subsets[(subsets >> u) & 1 == 0]
                k = sizes[without]
                weight = np.array([factorial(s) * factorial(m - s - 1) / factorial(m) for s in k])
                phi[feat] += (weight[:, None] * (v[without | (1 << u)] - v[without])).sum(axis=0)
        return phi * self.scale_out

    def explain(self, patient: dict, top: int = 8) -> list:
        row = self.clean(patient)
        x = self.transform(row)
        cls = int(np.argmax(self.predict_proba(patient)))
        phi = self.shap_values(x)[:, cls]
        totals = {}
        for field, value in zip(self.output_to_field, phi):
            totals[field] = totals.get(field, 0.0) + float(value)
        ranked = sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top]

        def show(v):
            if v is None:
                return "missing"
            return f"{v:.4g}" if isinstance(v, float) else str(v)

        return [{"feature": pretty(f), "value": show(row[f]), "effect": e} for f, e in ranked]
