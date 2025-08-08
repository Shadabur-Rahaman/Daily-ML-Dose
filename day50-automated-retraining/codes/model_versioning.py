# model_versioning.py
import mlflow

def register_model(model_path, version_name):
    mlflow.log_artifact(model_path)
    mlflow.register_model(model_path, f"models:/{version_name}")

if __name__ == "__main__":
    register_model("model.pkl", "MyModel")
