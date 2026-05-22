import mlflow
import mlflow.sklearn
import json
import os
import shutil

mlflow.set_tracking_uri("sqlite:///mlflow.db")
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
        
        # update champion metrics file
        with open("champion_metrics.json", "w") as f:
            json.dump({"roc_auc": new_auc, "f1": new_f1}, f)
        
        # also set alias in MLflow registry
        versions = client.search_model_versions("name='f1-podium-champion'")
        latest = max(versions, key=lambda v: int(v.version)).version
        client.set_registered_model_alias("f1-podium-champion", "champion", latest)
        print(f"Version {latest} promoted to champion")


        shutil.copy("model/challenger.pkl", "model/champion.pkl")
        print("Champion model file updated")

        

except FileNotFoundError:
    print("\nNo champion yet — setting current model as champion")
    with open("champion_metrics.json", "w") as f:
        json.dump({"roc_auc": new_auc, "f1": new_f1}, f)
    
    
    # set alias for first time
    versions = client.search_model_versions("name='f1-podium-champion'")
    latest = max(versions, key=lambda v: int(v.version)).version
    client.set_registered_model_alias("f1-podium-champion", "champion", latest)
    print(f"Version {latest} set as first champion")
    shutil.copy("model/challenger.pkl", "model/champion.pkl")
    print("First champion saved")