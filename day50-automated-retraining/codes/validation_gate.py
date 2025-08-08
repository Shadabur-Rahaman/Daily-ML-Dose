# validation_gate.py
from utils import evaluate_model, load_model

def validate_new_model():
    model = load_model("new_model.pkl")
    metrics = evaluate_model(model)
    if metrics['accuracy'] > 0.9:
        print("✅ Model passed validation.")
        return True
    print("❌ Model failed validation.")
    return False
