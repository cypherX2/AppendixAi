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
├── streamlit_app.py          ← Frontend: the website (form, results, charts)
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
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Then open http://localhost:8501.

## Dataset

Regensburg Pediatric Appendicitis Dataset (Kaggle / original authors Marcinkevičs et al.).
Only 9 anonymised sample rows are included in this repository; the full dataset is not redistributed here.

## Limitations

- Single-centre retrospective data, no external validation.
- "Complicated" is the dataset's severity label; it covers advanced disease such as perforation but is
  not a direct perforation test.
- Raw ultrasound images were studied (ResNet18) but did not improve accuracy, so the app uses the
  structured ultrasound report.
