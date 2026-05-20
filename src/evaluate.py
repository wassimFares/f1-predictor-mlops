import mlflow
import json

mlflow.set_tracking_uri("mlruns")
client = mlflow.tracking.MlflowClient()

# load new model metrics
with open("metrics.json") as f:
    new_metrics = json.load(f)

new_auc = new_metrics["roc_auc"]
new_f1 = new_metrics["f1"]
print(f"Challenger — ROC-AUC: {new_auc:.4f} | F1: {new_f1:.4f}")

# load champion metrics
try:
    champion = client.get_model_version_by_alias("f1-podium-champion", "champion")
    champion_run = client.get_run(champion.run_id)
    champion_auc = float(champion_run.data.metrics["roc_auc"])
    champion_f1 = float(champion_run.data.metrics["f1"])
    print(f"Champion   — ROC-AUC: {champion_auc:.4f} | F1: {champion_f1:.4f}")

    # require meaningful improvement — not just noise
    if new_auc > champion_auc + 0.005:
        print("\nChallenger wins — promoting to champion")

        # get the latest registered version (just trained)
        all_versions = client.search_model_versions("name='f1-podium-champion'")
        latest_version = max(all_versions, key=lambda v: int(v.version)).version

        # promote new champion
        client.set_registered_model_alias(
            "f1-podium-champion", "champion", latest_version
        )
        print(f"Version {latest_version} is now champion")

    else:
        print("\nChampion holds — no promotion")
        print(f"Needed improvement: >0.005 ROC-AUC")
        print(f"Actual difference:   {new_auc - champion_auc:+.4f}")

except Exception as e:
    print(f"\nNo champion found — promoting first model: {e}")
    all_versions = client.search_model_versions("name='f1-podium-champion'")
    latest_version = max(all_versions, key=lambda v: int(v.version)).version
    client.set_registered_model_alias(
        "f1-podium-champion", "champion", latest_version
    )
    print(f"Version {latest_version} set as first champion")