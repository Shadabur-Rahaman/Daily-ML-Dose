from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define weak learner
weak_learner = DecisionTreeClassifier(max_depth=1)

# AdaBoost
model = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=100, learning_rate=0.5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred))
