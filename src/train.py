import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json
import yaml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import os
import pickle

mlflow.set_tracking_uri("sqlite:///mlflow.db")

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

df = pd.read_csv("data/processed/features.csv")
df = df.sort_values(["season","round"]).reset_index(drop=True)

feature_cols = [
    "grid", "gap_to_pole", "best_quali_time",
    "rolling_podium_rate", "rolling_points",
    "team_rolling_points", "circuit_podium_rate",
    "avg_temp", "avg_humidity", "avg_wind", "rainfall"
]

X = df[feature_cols]
y = df["podium"]

mlflow.set_experiment("f1-podium-predictor")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.set_tag("model_type", "xgboost_classifier")
    mlflow.set_tag("eval_strategy", "walk_forward_5fold")

    # walk-forward cross validation
    tscv = TimeSeriesSplit(n_splits=5)
    f1_scores, auc_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                scale_pos_weight=int((y_train==0).sum()/(y_train==1).sum()),
                random_state=params["random_state"],
                eval_metric="logloss"
            ))
        ])
        fold_model.fit(X_train, y_train)
        preds = fold_model.predict(X_test)
        probs = fold_model.predict_proba(X_test)[:,1]

        f1_scores.append(f1_score(y_test, preds))
        auc_scores.append(roc_auc_score(y_test, probs))
        print(f"Fold {fold+1} — F1: {f1_scores[-1]:.4f} | AUC: {auc_scores[-1]:.4f}")

    mean_f1 = np.mean(f1_scores)
    mean_auc = np.mean(auc_scores)

    mlflow.log_metric("f1", round(mean_f1, 4))
    mlflow.log_metric("roc_auc", round(mean_auc, 4))

    print(f"\nMean F1:      {mean_f1:.4f}")
    print(f"Mean ROC-AUC: {mean_auc:.4f}")

    # final model trained on ALL data — no holdout
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            scale_pos_weight=int((y==0).sum()/(y==1).sum()),
            random_state=params["random_state"],
            eval_metric="logloss"
        ))
    ])
    final_model.fit(X, y)

    mlflow.sklearn.log_model(
        final_model, "model",
        registered_model_name="f1-podium-champion"
    )

    with open("metrics.json", "w") as f:
        json.dump({
            "f1": round(mean_f1, 4),
            "roc_auc": round(mean_auc, 4)
        }, f)

    

    os.makedirs("model", exist_ok=True)
    with open("model/challenger.pkl", "wb") as f:
        pickle.dump(final_model, f)
    print("Challenger saved to model/challenger.pkl")