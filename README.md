# 🏎️ F1 Podium Predictor — End-to-End MLOps Pipeline

Predicts F1 podium finishes using qualifying telemetry, historical driver form, and weather data. Built as a production-grade MLOps pipeline with automated retraining, experiment tracking, champion/challenger model promotion, and a REST API served in Docker.

---

## Architecture

```
Every Monday (GitHub Actions)
        ↓
collect_latest.py   → fetches latest race from FastF1
        ↓
features.py         → rolling form, circuit history, qualifying features
        ↓
train.py            → walk-forward CV, logs to MLflow (SQLite)
        ↓
evaluate.py         → champion vs challenger (promotes if ROC-AUC improves >0.005)
        ↓
FastAPI + Docker    → serves predictions via REST API
```

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC (walk-forward CV) | **0.926** |
| F1 Score | **0.665** |
| Evaluation strategy | 5-fold time-series split |
| Training data | 2022–2026 (1900+ driver-race rows) |

---

## Features Used

| Feature | Source | Why it matters |
|---|---|---|
| `grid` | Qualifying | Strongest predictor — P1 has ~80% podium rate |
| `gap_to_pole` | Qualifying | Raw car pace this weekend |
| `best_quali_time` | Qualifying | Absolute lap time |
| `rolling_podium_rate` | Race history | Driver form last 5 races |
| `rolling_points` | Race history | Points scoring consistency |
| `team_rolling_points` | Race history | Constructor momentum |
| `circuit_podium_rate` | Race history | Track-specific performance |
| `avg_temp/humidity/wind` | Weather | Race conditions |
| `rainfall` | Weather | Wet race flag |

---

## Stack

| Component | Tool |
|---|---|
| Data collection | FastF1 |
| Feature engineering | Pandas, NumPy |
| Model | XGBoost (GBT classifier) |
| Experiment tracking | MLflow (SQLite backend) |
| CI/CD | GitHub Actions |
| Serving | FastAPI + Uvicorn |
| Containerization | Docker |

---

## Project Structure

```
f1-predictor-mlops/
├── .github/
│   └── workflows/
│       └── retrain.yaml      ← automated Monday retraining
├── app/
│   └── main.py               ← FastAPI endpoints (pre/post qualifying)
├── data/
│   ├── raw/
│   │   ├── races.csv
│   │   ├── qualifying.csv
│   │   └── weather.csv
│   └── processed/
│       └── features.csv
├── model/
│   ├── challenger.pkl        ← latest trained model
│   └── champion.pkl          ← promoted champion model
├── monitoring/               ← Evidently drift reports (future)
├── notebooks/
│   ├── EDA.ipynb
│   └── feature_plots.png
├── src/
│   ├── collect.py            ← full historical data collection
│   ├── collect_latest.py     ← incremental race collection (CI)
│   ├── features.py           ← feature engineering pipeline
│   ├── train.py              ← walk-forward CV + MLflow logging
│   └── evaluate.py           ← champion/challenger promotion
├── .gitignore
├── champion_metrics.json     ← persisted champion score
├── Dockerfile
├── metrics.json              ← latest challenger metrics
├── mlflow.db                 ← MLflow experiment tracking (SQLite)
├── params.yaml               ← model hyperparameters
├── requirements.txt
└── README.md
```

---

## API Endpoints

### Pre-qualifying prediction
```
GET /predict/pre-quali?season=2026&round=5
```
Uses only historical form and circuit data. Available before Saturday qualifying. Lower accuracy — grid positions unknown.

### Post-qualifying prediction
```
GET /predict/post-quali?season=2026&round=5
```
Fetches qualifying results automatically from FastF1. Full feature set. Run after Saturday qualifying for best accuracy.

### Example response
```json
{
  "race": "2026 Round 1 — Australian Grand Prix",
  "stage": "post-qualifying",
  "predictions": [
    {
      "driver": "RUS",
      "team": "Mercedes",
      "grid": 1,
      "podium_probability": 0.9256,
      "predicted_podium": true
    },
    {
      "driver": "ANT",
      "team": "Mercedes",
      "grid": 2,
      "podium_probability": 0.8561,
      "predicted_podium": true
    },
    {
      "driver": "LEC",
      "team": "Ferrari",
      "grid": 4,
      "podium_probability": 0.4950,
      "predicted_podium": false
    }
  ]
}
```

---

## Running Locally

```bash
# install dependencies
pip install -r requirements.txt

# collect historical data (one-time)
python src/collect.py

# build features
python src/features.py

# train and log to MLflow
python src/train.py

# view experiment dashboard
mlflow ui --backend-store-uri sqlite:///mlflow.db

# start API
uvicorn app.main:app --reload
# → http://localhost:8000/docs
```

---

## Running with Docker

```bash
docker build -t f1-predictor .
docker run -p 8000:8000 f1-predictor
# → http://localhost:8000/docs
```

---

## Automated Retraining

GitHub Actions triggers every Thursday at 8am UTC after race weekend:

1. `collect_latest.py` fetches the most recent race(s) — loops until no new data
2. `features.py` rebuilds the feature set with rolling windows
3. `train.py` retrains with walk-forward cross-validation, logs to MLflow
4. `evaluate.py` compares new model vs champion — promotes only if ROC-AUC improves by >0.005
5. Updated CSVs, `mlflow.db`, and `champion.pkl` committed back to repo

The pipeline handles season rollovers automatically and skips gracefully when no new race data is available.

---

## Key Design Decisions

**Walk-forward cross-validation** — standard train/test split leaks future race data into training. Walk-forward CV evaluates each fold on unseen future races, giving an honest performance estimate.

**Temporal data collection** — F1 timing API has inconsistent availability from cloud VMs. Collection runs where API access is reliable; pushing new data triggers automated retraining.

**Champion/challenger with threshold** — model is only promoted if ROC-AUC improves by >0.005, preventing promotion on statistical noise.

**Grid penalty awareness** — cases like Verstappen qualifying P1 but starting P14 due to engine penalties are correctly represented: `gap_to_pole=0.0` with `grid=14`. The model learns this combination means "penalized fast car."

---
