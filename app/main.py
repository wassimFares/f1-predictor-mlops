import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import fastf1
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import pickle

os.makedirs("data/cache", exist_ok=True)
fastf1.Cache.enable_cache("data/cache")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
fastf1.Cache.enable_cache("data/cache")

app = FastAPI(
    title="F1 Podium Predictor",
    description="Predicts F1 podium finishes — pre-qualifying (form only) and post-qualifying (full features)",
    version="1.0.0"
)

# load champion model
try:
    with open("model/champion.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Model load failed: {e}")
    model = None

FEATURE_COLS = [
    "grid", "gap_to_pole", "best_quali_time",
    "rolling_podium_rate", "rolling_points",
    "team_rolling_points", "circuit_podium_rate",
    "avg_temp", "avg_humidity", "avg_wind", "rainfall"
]

def get_rolling_features(driver, team, circuit, season, round_num):
    df = pd.read_csv("data/processed/features.csv")
    df = df.sort_values(["season", "round"])

    # filter past races only
    past = df[
        (df["season"] < season) |
        ((df["season"] == season) & (df["round"] < round_num))
    ]

    # get driver's most recent row — rolling features already computed
    driver_past = past[past["driver"] == driver]
    
    if len(driver_past) == 0:
        return {
            "rolling_podium_rate": float(df["rolling_podium_rate"].median()),
            "rolling_points": float(df["rolling_points"].median()),
            "team_rolling_points": float(df["team_rolling_points"].median()),
            "circuit_podium_rate": float(df["circuit_podium_rate"].median()),
        }
    
    latest_row = driver_past.iloc[-1]
    
    # circuit podium rate — compute from past races at this circuit
    circuit_races = past[
        (past["driver"] == driver) & (past["circuit"] == circuit)
    ]
    circuit_podium_rate = circuit_races["podium"].mean() if len(circuit_races) > 0 else 0.0

    return {
        "rolling_podium_rate": round(float(latest_row["rolling_podium_rate"]), 4),
        "rolling_points": round(float(latest_row["rolling_points"]), 4),
        "team_rolling_points": round(float(latest_row["team_rolling_points"]), 4),
        "circuit_podium_rate": round(float(circuit_podium_rate), 4),
    }

def get_weather(season, round_num):
    """fetch weather from FastF1 or use defaults"""
    try:
        session = fastf1.get_session(season, round_num, "R")
        session.load(telemetry=False, weather=True, messages=False, laps=False)
        w = session.weather_data
        return {
            "avg_temp": round(float(w["AirTemp"].mean()), 2),
            "avg_humidity": round(float(w["Humidity"].mean()), 2),
            "avg_wind": round(float(w["WindSpeed"].mean()), 2),
            "rainfall": int(w["Rainfall"].any()),
        }
    except:
        return {"avg_temp": 22.0, "avg_humidity": 50.0, "avg_wind": 10.0, "rainfall": 0}

def get_qualifying(season, round_num):
    session = fastf1.get_session(season, round_num, "Q")
    session.load(telemetry=False, weather=False, messages=False, laps=False)

    results = session.results
    pole_time = None

    rows = []
    for _, row in results.iterrows():
        best_time = None
        for q in ["Q3", "Q2", "Q1"]:
            t = row[q]
            if pd.notna(t):
                best_time = t.total_seconds()
                break

        if pole_time is None and best_time is not None:
            pole_time = best_time

        # safely convert position
        position = pd.to_numeric(row["Position"], errors="coerce")
        if pd.isna(position):
            continue  # skip drivers with no position

        rows.append({
            "driver": row["Abbreviation"],
            "team": row["TeamName"],
            "grid": int(position),
            "best_quali_time": best_time if best_time else 90.0,
            "gap_to_pole": (best_time - pole_time) 
                           if best_time and pole_time else 2.0,
        })

    return rows, session.event["EventName"]

def make_predictions(drivers_data, season, round_num, circuit):
    """run model inference and return sorted predictions"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rows = []
    meta = []

    for d in drivers_data:
        rolling = get_rolling_features(
            d["driver"], d["team"], circuit, season, round_num
        )
        rows.append({
            "grid": d.get("grid", 10),
            "gap_to_pole": d.get("gap_to_pole", 1.5),
            "best_quali_time": d.get("best_quali_time", 90.0),
            **rolling,
            **d.get("weather", {"avg_temp": 22.0, "avg_humidity": 50.0,
                                 "avg_wind": 10.0, "rainfall": 0})
        })
        meta.append({"driver": d["driver"], "team": d["team"],
                     "grid": d.get("grid", 10)})

    X = pd.DataFrame(rows)[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]

    predictions = []
    for i, m in enumerate(meta):
        predictions.append({
            **m,
            "podium_probability": round(float(probs[i]), 4),
            "predicted_podium": bool(probs[i] >= 0.5)
        })

    return sorted(predictions, key=lambda x: -x["podium_probability"])

# ── endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "F1 Podium Predictor API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/predict/pre-quali")
def predict_pre_quali(season: int, round: int):
    """
    Pre-qualifying prediction — uses only historical form and circuit data.
    No qualifying results needed. Lower accuracy but available before Saturday.
    """
    try:
        # get driver list from most recent race
        df = pd.read_csv("data/processed/features.csv")
        print(f"CSV loaded: {len(df)} rows")
        
        latest = df[df["season"] == df["season"].max()]
        latest = latest[latest["round"] == latest["round"].max()]
        drivers = latest[["driver","team"]].drop_duplicates().to_dict("records")
        print(f"Drivers found: {len(drivers)}")

        # get circuit name from FastF1
        try:
            session = fastf1.get_session(season, round, "R")
            session.load(telemetry=False, weather=False, messages=False, laps=False)
            circuit = session.event["EventName"]
        except:
            circuit = "Unknown Circuit"

        weather = get_weather(season, round)

        drivers_data = []
        for d in drivers:
            drivers_data.append({
                "driver": d["driver"],
                "team": d["team"],
                "grid": 10,           # unknown pre-quali → use midfield default
                "gap_to_pole": 1.0,   # unknown pre-quali → use median default
                "best_quali_time": 90.0,
                "weather": weather
            })

        predictions = make_predictions(drivers_data, season, round, circuit)

        return {
            "race": f"{season} Round {round} — {circuit}",
            "stage": "pre-qualifying",
            "warning": "Grid positions unknown — predictions based on form only",
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predict/post-quali")
def predict_post_quali(season: int, round: int):
    """
    Post-qualifying prediction — fetches qualifying results automatically.
    Uses full feature set. Higher accuracy. Run after Saturday qualifying.
    """
    try:
        quali_results, circuit = get_qualifying(season, round)
        weather = get_weather(season, round)

        drivers_data = []
        for d in quali_results:
            drivers_data.append({
                **d,
                "weather": weather
            })

        predictions = make_predictions(drivers_data, season, round, circuit)

        return {
            "race": f"{season} Round {round} — {circuit}",
            "stage": "post-qualifying",
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))