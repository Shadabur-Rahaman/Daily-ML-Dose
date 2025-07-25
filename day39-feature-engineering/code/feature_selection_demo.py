# Feature Selection using correlation and embedded methods
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Correlation-based feature selection
correlations = X.corrwith(y).abs()
selected_by_corr = correlations[correlations > 0.2].index.tolist()
print("Correlation-selected features:\n", selected_by_corr)

# Embedded method: Feature importance from RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=X.columns)
selected_by_rf = importances.sort_values(ascending=False).head(10)
print("\nTop 10 Features by Random Forest:\n", selected_by_rf)
