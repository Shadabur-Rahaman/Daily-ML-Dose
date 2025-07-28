from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestClassifier().fit(X_train, y_train)

# Permutation Importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# Plot
features = load_wine().feature_names
importances = result.importances_mean

plt.barh(features, importances)
plt.title("Permutation Feature Importance")
plt.xlabel("Importance")
plt.show()
