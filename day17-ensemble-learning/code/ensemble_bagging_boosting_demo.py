# ensemble_bagging_boosting_demo.py

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Bagging - Random Forest
bag_model = RandomForestClassifier(n_estimators=100, random_state=42)
bag_model.fit(X_train, y_train)
y_pred_bag = bag_model.predict(X_test)

# Boosting - AdaBoost
boost_model = AdaBoostClassifier(n_estimators=100, random_state=42)
boost_model.fit(X_train, y_train)
y_pred_boost = boost_model.predict(X_test)

# Boosting - Gradient Boosting
grad_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
grad_model.fit(X_train, y_train)
y_pred_grad = grad_model.predict(X_test)

# Accuracy
print(f"Random Forest Accuracy (Bagging): {accuracy_score(y_test, y_pred_bag):.3f}")
print(f"AdaBoost Accuracy (Boosting): {accuracy_score(y_test, y_pred_boost):.3f}")
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, y_pred_grad):.3f}")
