from catboost import CatBoostClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset with categorical features
data = fetch_openml(name="adult", version=2, as_frame=True)
X = data.data.select_dtypes(include=[object, 'category']).fillna("missing").copy()
y = data.target

# Encode target labels
y = y.astype("category").cat.codes

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical feature indices
cat_features = list(range(X.shape[1]))

# CatBoost
model = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.1, verbose=0)
model.fit(X_train, y_train, cat_features=cat_features)

# Evaluate
y_pred = model.predict(X_test)
print("CatBoost Accuracy:", accuracy_score(y_test, y_pred))
