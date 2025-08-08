# retrain_pipeline.py
from sklearn.ensemble import RandomForestClassifier
from utils import load_data, save_model

def retrain_model():
    X_train, y_train, X_test, y_test = load_data()
    model = RandomForestClassifier().fit(X_train, y_train)
    save_model(model)
    print("✅ Model retrained and saved.")

if __name__ == "__main__":
    retrain_model()
