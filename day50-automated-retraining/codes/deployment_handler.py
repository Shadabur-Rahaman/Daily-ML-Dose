# deployment_handler.py
import subprocess

def deploy_model(model_path):
    subprocess.run(["deploy_service", "--model", model_path])
    print(f"🚀 Deployed {model_path}")

if __name__ == "__main__":
    deploy_model("model.pkl")
