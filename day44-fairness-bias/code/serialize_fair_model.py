import joblib
joblib.dump(fair_model, 'fair_model.pkl')
model = joblib.load('fair_model.pkl')
