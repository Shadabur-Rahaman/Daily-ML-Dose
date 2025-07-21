from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train base models
rf = RandomForestClassifier(n_estimators=50, random_state=42)
catboost = CatBoostClassifier(verbose=0)
rf.fit(X_train, y_train)
catboost.fit(X_train, y_train)

# Predict probabilities
rf_preds = rf.predict_proba(X_test)
cat_preds = catboost.predict_proba(X_test)

# Blend predictions (average)
blended_preds = (rf_preds + cat_preds) / 2
final_preds = np.argmax(blended_preds, axis=1)

# Accuracy
print("Blended Accuracy:", accuracy_score(y_test, final_preds))
