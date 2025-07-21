from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Voting Classifier - Hard and Soft
log_clf = LogisticRegression()
dt_clf = DecisionTreeClassifier()
svm_clf = SVC(probability=True)

# Hard Voting
voting_hard = VotingClassifier(
    estimators=[('lr', log_clf), ('dt', dt_clf), ('svc', svm_clf)],
    voting='hard'
)

# Soft Voting
voting_soft = VotingClassifier(
    estimators=[('lr', log_clf), ('dt', dt_clf), ('svc', svm_clf)],
    voting='soft'
)

# Fit and predict
voting_hard.fit(X_train, y_train)
voting_soft.fit(X_train, y_train)

hard_preds = voting_hard.predict(X_test)
soft_preds = voting_soft.predict(X_test)

# Accuracy
print("Hard Voting Accuracy:", accuracy_score(y_test, hard_preds))
print("Soft Voting Accuracy:", accuracy_score(y_test, soft_preds))
