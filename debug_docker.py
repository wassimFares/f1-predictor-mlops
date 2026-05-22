import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:////app/mlflow.db")
client = mlflow.tracking.MlflowClient()

try:
    version = client.get_model_version_by_alias("f1-podium-champion", "champion")
    print("Champion version:", version.version)
    print("Source:", version.source)
    
    # try loading
    model = mlflow.sklearn.load_model(f"models:/f1-podium-champion/{version.version}")
    print("Model loaded successfully:", type(model))
except Exception as e:
    print("Error:", e)
    
    # try loading directly by source
    try:
        model = mlflow.sklearn.load_model(version.source)
        print("Loaded by source successfully")
    except Exception as e2:
        print("Source load failed:", e2)