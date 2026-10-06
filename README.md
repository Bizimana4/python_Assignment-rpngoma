# Crop Recommendation with a Feed-Forward Neural Network (FNN)

CAT 2 Practical - ITLPA701 Python and Fundamentals of AI (Rwanda Polytechnic, Ngoma College)

Recommends the best crop (22 classes) from soil nutrients (N, P, K), soil pH, temperature, humidity and rainfall.
Approach: deep learning (FNN, Keras/TensorFlow). Demo: Gradio web app.

## 1. Setup (one time)

The dataset (`data/Crop_recommendation.csv`) and a **trained model** (`models/`) are already included, so the project runs straight away.

### Quick start on Windows

If `python` is not recognized in PowerShell/Command Prompt, use the Windows launcher instead:

```
pip install -r requirements.txt
py -3 app.py
```

You can also double-click the included launcher files:

- `run_app.bat` — launches the Gradio app
- `train_model.bat` — retrains the model

The app usually opens at http://127.0.0.1:7862 or another local port shown in the terminal.

Use Python 3.9 - 3.12. TensorFlow is a large download, so install it before the exam if the lab internet is slow.

To retrain from scratch (about 1-2 minutes) and regenerate everything in `results/` and `models/`:
```
py -3 train.py
```

## Deploy from GitHub

The app can be hosted as a public web service on Render while this GitHub repository remains the source code:

1. Push the project to the `main` branch on GitHub.
2. Sign in to [Render](https://dashboard.render.com), choose **New + > Blueprint**, and connect this repository.
3. Render reads `render.yaml` and builds the web service. When deployment finishes, open the `onrender.com` URL shown in the Render dashboard.

The included configuration uses Render's free plan. Free services sleep after 15 minutes without traffic and can take about a minute to wake up when opened.

Or open **`CAT_notebook.ipynb`** in Jupyter / PyCharm and run all cells. It follows the marking scheme in order, shows the results and launches the app. (It already contains the outputs from a full run.)

If the saved model cannot be loaded because of a different TensorFlow version, the app retrains automatically and carries on.

## 2. Results of the included run (real dataset)

| Item | Value |
|---|---|
| Data | 2,200 rows, 22 crops x 100, no missing values, no duplicates |
| Split | 1,540 train / 330 validation / 330 test (stratified) |
| Selected model | `v1_baseline` - one hidden layer of 32 units, no dropout, Adam lr 0.01 |
| Test accuracy | **98.5%** (325 of 330 correct) |
| Macro precision / recall / F1 | 0.985 / 0.985 / 0.985 |
| Errors | 5 total: rice<->jute, lentil<->mothbeans, maize->cotton (crops with similar conditions) |
| Experiments | v1, v2 (deeper + dropout), v3 (engineered features) all within 1 validation sample of each other |

Honest finding: the engineered features and the larger networks did **not** improve accuracy, because the 7 raw measurements already separate the crops very well. A difference of one sample out of 330 is within normal noise, so the simplest adequate model was chosen. Exact numbers are in `results/experiments.csv` and `results/metrics.json`. Retraining on another machine can shift a figure by a sample or so.

## 3. Folder layout

```
config.json            all settings (seed, split, epochs, the 3 experiments)
train.py               full pipeline: load -> clean -> split -> scale -> train -> evaluate -> save
app.py                 Gradio interface (top-3 crops, confidence, out-of-range warnings)
CAT_notebook.ipynb     step-by-step notebook with explanations for every rubric row
src/data_prep.py       loading, quality report, cleaning, label encoding, split, scaling
src/features.py        feature engineering (6 extra features)
src/model.py           Keras FNN builder
src/evaluate.py        metrics, confusion matrix, training curves
data/                  Crop_recommendation.csv (included)
models/                (included; recreated by train.py) crop_fnn.keras, scaler.joblib, label_encoder.joblib, model_meta.json
results/               (included; recreated by train.py) experiments.csv, metrics.json, classification_report.txt, PNG plots
tools/make_synthetic_data.py   TEST ONLY - fake data to check the code runs
```

## 4. Where each marking-scheme row is covered

| Rubric row | Where |
|---|---|
| Environment functionalities specified / configured | notebook sections intro + 1, `requirements.txt`, `config.json` |
| Data acquired from the given source | notebook section 2, `src/data_prep.load_data` |
| Data pre-processed | notebook section 3, `src/data_prep.py` (quality report, cleaning, encoding, stratified split, scaling fit on train only) |
| Data features engineered | `src/features.py`, notebook section 4 |
| Model features engineered | notebook section 5 + `src/model.py` (layers, activations, dropout) |
| Approach and model justified | notebook section 5 |
| Parameters selected and explained | notebook section 5 table, `config.json` |
| Application implemented and tested | `train.py`, `app.py`, notebook sections 6 and 9 |
| Evaluation metrics chosen | accuracy, macro precision/recall/F1, confusion matrix (`src/evaluate.py`) |
| Evaluation implemented and interpreted | notebook section 7 (auto-interpretation + your own paragraph) |
| Parameters adjusted from results | 3 experiments v1 -> v2 -> v3 in `config.json`, compared in `results/experiments.csv` |
| Model/config saved for reproducibility | `models/`, `config.json`, fixed seed |
| Deployed / demonstrated | `app.py` (Gradio) and the notebook |
| Responsible use (task 8) | notebook section 10, plus warnings built into the app |

## 5. What was tested

All of the following were run successfully on the **real dataset** with TensorFlow 2.21, scikit-learn 1.8, pandas 3.0 and Gradio 6:
- `python train.py` end to end, and the notebook (all cells, run in a real Jupyter kernel) - both gave identical results.
- The Gradio app through its live endpoint: a rice-type input returns rice (97.9%), a chickpea-type input returns chickpea, and an out-of-range pH shows a warning.
- The recovery path: with a corrupted model file the app retrained itself and still answered.

The dataset file is a verified copy of the Kaggle *Crop Recommendation Dataset* (identical across several independent mirrors). You may replace it with your own Kaggle download under the same file name.

## 6. Tips for the exam day

- Write your own 3-4 line conclusion in notebook section 7 (accuracy, which crops get confused, over-fitting check).
- Be ready to explain why the simple model won: the raw features already separate the crops, and a 1-sample difference is noise.
- Be ready to explain: why scaling, why a validation set, why early stopping, and what the experiments showed.
- If the app shows **"SETUP PROBLEM: No trained model found"** (or a red *Error*), you have not trained yet: put the CSV in `data/` and run `python train.py` first, then restart `python app.py`.
- If a library is missing: `pip install <name>`. If port 7860 is busy, change the port in `app.py` (`launch(server_port=7870)`).
