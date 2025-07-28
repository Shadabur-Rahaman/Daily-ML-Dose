import shap
import xgboost as xgb
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
model = xgb.XGBClassifier().fit(X, y)

explainer = shap.Explainer(model)
shap_values = explainer(X)
shap.summary_plot(shap_values, X)
