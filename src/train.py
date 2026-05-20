import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

df = pd.read_csv("data/processed/features.csv")

feature_cols = [
    "grid", "gap_to_pole", "best_quali_time",
    "rolling_podium_rate", "rolling_points",
    "team_rolling_points", "circuit_podium_rate",
    "avg_temp", "avg_humidity", "avg_wind", "rainfall"
]

X = df[feature_cols]
y = df["podium"]

# split by season to avoid leakage — train on 2022-2024, test on 2025 and 2026
X_train = X[df["season"] < 2025]
y_train = y[df["season"] < 2025]
X_test = X[df["season"].isin([2025, 2026])]
y_test = y[df["season"].isin([2025, 2026])]

print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
print(f"Train podium rate: {y_train.mean():.2%}")
print(f"Test podium rate: {y_test.mean():.2%}")

mlflow.set_experiment("f1-podium-predictor")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.set_tag("model_type", "xgboost_classifier")
    mlflow.set_tag("split", "temporal_2025-2026_holdout")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            scale_pos_weight=int((y_train==0).sum() / (y_train==1).sum()),
            random_state=params["random_state"],
            eval_metric="logloss"
        ))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(model, "model",
                             registered_model_name="f1-podium-champion")

    with open("metrics.json", "w") as f:
        json.dump({"f1": round(f1, 4), "roc_auc": round(auc, 4)}, f)

    print(f"\nF1:      {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"\nscale_pos_weight: {int((y_train==0).sum() / (y_train==1).sum())}")
    print(f"\n{classification_report(y_test, preds, target_names=['no podium','podium'])}")