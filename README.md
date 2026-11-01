# 🩺 AppendixAi — AI Decision Support for Pediatric Appendicitis

A research web app that estimates, for a child with abdominal pain:

1. **Is it appendicitis?** (yes / no)
2. **If yes, is it complicated?** (the severity group that includes perforated appendicitis)

> ⚠️ **Research prototype — not a medical device.** Predictions must never replace clinical judgement.

## Results (156 locked test patients, never used for training or model selection)

| Metric | Value |
|---|---|
| Accuracy | **85.9%** (95% CI 80.1–91.0%) |
| Macro ROC-AUC | 0.94 |
| Appendicitis sensitivity / specificity | 95.7% / 90.5% |
| Complicated appendicitis sensitivity / specificity | 58.3% / 97.0% |

Model: tuned **CatBoost**, chosen by 5-fold cross-validation among XGBoost, LightGBM, CatBoost,
Random Forest and two ensembles. Inputs: 32 routine variables (demographics, symptoms, examination,
blood/urine tests, structured ultrasound report).

## Project structure

```
appendicitis-ai/
├── streamlit_app.py          ← Website version 1: Streamlit
├── web_app.py                ← Website version 2: Flask server for the HTML/CSS/JS site
├── web/
│   ├── index.html            ← page structure
│   ├── style.css             ← design
│   └── app.js                ← behaviour (form, samples, results, report download)
├── predict.py                ← predictions from the command line (no website)
├── backend/
│   └── predictor.py          ← Backend: loads the model, validates input, predicts, explains
├── model/
│   ├── appendicitis_model.joblib   ← trained model (preprocessing + CatBoost in one pipeline)
│   └── model_info.json             ← input fields, allowed values, ranges, test metrics
├── data/sample_patients.csv  ← 9 real, anonymised test patients for the demo
├── figures/                  ← evaluation graphs shown in the app
├── notebooks/                ← full analysis notebook with all outputs
├── requirements.txt          ← exact package versions
└── .streamlit/config.toml    ← theme
```

## Run on your computer

```bash
pip install -r requirements-full.txt
```

- `requirements.txt` → only Flask + NumPy: what the website needs (used by Vercel).
- `requirements-full.txt` → everything (Streamlit, CatBoost, notebooks, retraining).

**HTML/CSS/JavaScript website** (works fully offline):
```bash
python web_app.py
```
Then open http://localhost:5000.

**Streamlit website:**
```bash
streamlit run streamlit_app.py
```
Then open http://localhost:8501.

Both websites use the same model and give identical predictions.

## Deploy on Vercel

The HTML website deploys to Vercel as a Flask app (`app.py`). The model is exported to
`model/lite_model.json` and run with NumPy only (`backend/lite.py`), so the deployment is ~80 MB
instead of >500 MB. Predictions and explanations match the original CatBoost model exactly.
After retraining, re-export with `python tools/export_lite_model.py`.

## Dataset

Regensburg Pediatric Appendicitis Dataset (Kaggle / original authors Marcinkevičs et al.).
Only 9 anonymised sample rows are included in this repository; the full dataset is not redistributed here.

## Limitations

- Single-centre retrospective data, no external validation.
- "Complicated" is the dataset's severity label; it covers advanced disease such as perforation but is
  not a direct perforation test.
- Raw ultrasound images were studied (ResNet18) but did not improve accuracy, so the app uses the
  structured ultrasound report.
