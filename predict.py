"""Run the appendicitis model without the website.

Usage (from the appendicitis-ai folder):
    python predict.py              -> predicts the 9 sample patients
    python predict.py my.csv       -> predicts every row of your own CSV (same column names)
"""
import sys

import pandas as pd

from backend import AppendicitisPredictor

sys.stdout.reconfigure(encoding="utf-8")   # lets the Windows terminal print ✅ / ⚠
predictor = AppendicitisPredictor()


def show(name, patient, true_label=None):
    result = predictor.predict(patient)
    print(f"\n=== {name} ===")
    print(f"Diagnosis        : {result['diagnosis']} (P = {result['appendicitis_probability']:.1%})")
    print(f"Most likely class: {result['predicted_class']}")
    print(f"Complicated risk : {result['complicated_risk']} (P = {result['complicated_probability']:.1%})")
    if true_label:
        print(f"True diagnosis   : {true_label}  {'✅' if true_label == result['predicted_class'] else '❌'}")
    for warning in predictor.validate(patient):
        print("  ⚠", warning)
    print("Top reasons:")
    print(predictor.explain(patient, top=5).to_string(index=False))


# 1) A patient typed in by hand — fill in what you know, leave the rest out
my_patient = {
    "Age": 10, "Sex": "male", "Body_Temperature": 38.2,
    "WBC_Count": 16.5, "CRP": 45, "Neutrophil_Percentage": 82,
    "Lower_Right_Abd_Pain": "yes", "Migratory_Pain": "yes", "Nausea": "yes",
    "Ipsilateral_Rebound_Tenderness": "yes",
    "US_Performed": "yes", "Appendix_on_US": "yes", "Appendix_Diameter": 9.5, "Free_Fluids": "yes",
}
show("Hand-entered patient", my_patient)

# 2) Patients from a CSV file (default: the 9 sample patients)
csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_patients.csv"
table = pd.read_csv(csv_path)
for i, row in table.iterrows():
    patient = {f: (None if pd.isna(row.get(f)) else row.get(f)) for f in predictor.features}
    show(row.get("sample_id", f"Row {i + 1}"), patient, row.get("true_diagnosis"))
