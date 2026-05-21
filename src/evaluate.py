import mlflow
import mlflow.sklearn
import json
import os

mlflow.set_tracking_uri("mlruns")
client = mlflow.tracking.MlflowClient()

with open("metrics.json") as f:
    new_metrics = json.load(f)

new_auc = new_metrics["roc_auc"]
new_f1 = new_metrics["f1"]
print(f"Challenger — ROC-AUC: {new_auc:.4f} | F1: {new_f1:.4f}")

# load champion from persistent JSON file
try:
    with open("champion_metrics.json") as f:
        champion_data = json.load(f)
    champion_auc = champion_data["roc_auc"]
    champion_f1 = champion_data["f1"]
    print(f"Champion   — ROC-AUC: {champion_auc:.4f} | F1: {champion_f1:.4f}")

    if new_auc > champion_auc + 0.005:
        print("\nChallenger wins — promoting to champion")
        with open("champion_metrics.json", "w") as f:
            json.dump({"roc_auc": new_auc, "f1": new_f1}, f)
        print(f"New champion: ROC-AUC {new_auc:.4f}")
    else:
        print("\nChampion holds — no promotion")
        print(f"Needed: >0.005 improvement | Actual: {new_auc - champion_auc:+.4f}")

except FileNotFoundError:
    print("\nNo champion yet — setting current model as champion")
    with open("champion_metrics.json", "w") as f:
        json.dump({"roc_auc": new_auc, "f1": new_f1}, f)
    print(f"Champion set: ROC-AUC {new_auc:.4f}")