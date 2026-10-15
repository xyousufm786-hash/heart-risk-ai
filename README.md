# Trustworthy Heart-Disease Risk Prediction

University AI project (healthcare). A statistically rigorous benchmark of nine models on the Cleveland heart-disease data, with calibration, conformal prediction (abstention), validated SHAP explanations, a fairness audit, robustness/shift tests, a federated-learning simulation (FedAvg vs FedProx) and a Streamlit app.

![architecture](results/figures/architecture.png)

## Headline results (Cleveland, n = 302, nested 5x5 CV)

| Finding | Number |
|---|---|
| Best model: logistic regression | ROC-AUC 0.919 ± 0.037, accuracy 0.854, recall 0.794 |
| Any complex model significantly better? | **No** (0 of 7 after Holm; 0 of 28 McNemar pairs) |
| Conformal coverage at 90% target | 91.0%; commits on 78.6% of patients at 89.2% accuracy |
| Federated (FedAvg, 5 hospitals, alpha = 0.1) | AUC 0.9205 vs 0.9203 centralised; FedProx no significant gain |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py                                # the saved model is included
pytest -q                                               # 9 tests
```

## Reproduce every result (about 15-25 min on a laptop CPU)

```bash
python scripts/00_data_audit.py                         # label inversion, placeholder codes, duplicates
python scripts/01_benchmark.py heart 5                  # nested CV, 9 models
python scripts/02_analysis.py heart                     # tables, significance tests, ROC/box plots
python scripts/01_benchmark.py pima 3 && python scripts/02_analysis.py pima
python scripts/03_trust.py                              # calibration, conformal, SHAP, fairness, shift
python scripts/04_federated.py                          # FedAvg / FedProx simulation
python scripts/05_train_final.py                        # saves models/heart_model.joblib
python scripts/07_architecture_diagram.py
python scripts/06_screenshots.py                        # needs the app running + Playwright Chromium
```

Seeds are fixed (42), so numbers should reproduce exactly on the same library versions (see `requirements.txt`; results here were produced with scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0, SHAP 0.53).

## Structure

```
data/raw/          Cleveland (two identical mirrors) and Pima CSVs
src/hrisk/         data, models, stats, trust, federated, predictor modules
scripts/           numbered, re-runnable experiment scripts
app/app.py         Streamlit user interface
models/            heart_model.joblib (logistic regression + conformal thresholds)
results/           all tables (CSV/JSON) and figures
docs/              screenshots and data card
tests/             unit tests
```

## Data

* **Cleveland Heart Disease** (UCI; Detrano et al., 1989). The CSV mirror's `target` column is **inverted** (1 = no disease); the loader sets `disease = 1 - target`. `ca = 4` and `thal = 0` are placeholders for missing values; one duplicate row is dropped. Evidence: `results/data_audit.json`.
* **Pima Indians Diabetes** (Smith et al., 1988): zeros in glucose/BP/skin/insulin/BMI treated as missing. Used only as a second benchmark.
* Source mirrors used to obtain the files: GitHub repositories `kb22/Heart-Disease-Prediction`, `sharmaroshan/Heart-UCI-Dataset`, `jbrownlee/Datasets`. Original source: https://archive.ics.uci.edu/

## Limitations

Single-centre 1980s data, 302 patients, no external validation; federated "hospitals" are simulated by label-skew partitioning; subgroup sizes are small. **Educational prototype - not a medical device.**

## Licence and AI-use note

Code: MIT. Data remain under their original UCI / NIDDK terms. State any AI-assistance used according to your institution's academic-integrity rules.

## App screenshots

| Low risk | Moderate (abstains) | High risk |
|---|---|---|
| ![A](docs/screenshots/case_A_low.png) | ![B](docs/screenshots/case_B_mixed.png) | ![C](docs/screenshots/case_C_high.png) |

## Disclaimer and acknowledgements

Educational prototype, **not a medical device**. Trained on a single 1980s hospital dataset (302 patients); not validated for clinical use.
The project was developed with AI assistance (Claude by Anthropic) for code, analysis and drafting; all results were run and checked by the author.

## License

MIT (see `LICENSE`). The datasets keep their original licences/terms (UCI repository and the Pima dataset).
