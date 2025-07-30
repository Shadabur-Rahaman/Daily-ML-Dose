from fairlearn.metrics import demographic_parity_difference
from sklearn.metrics import accuracy_score

dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=gender)
print(f"Demographic Parity Difference: {dp_diff:.3f}")
