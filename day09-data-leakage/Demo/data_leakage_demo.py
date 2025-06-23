```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ❌ LEAKED Preprocessing
scaler_leak = StandardScaler()
X_leak = scaler_leak.fit_transform(X)
X_train_leak = X_leak[:len(X_train)]
X_test_leak = X_leak[len(X_train):]

model = LogisticRegression(max_iter=1000)
model.fit(X_train_leak, y_train)
pred_leak = model.predict(X_test_leak)
print("Leaked Accuracy:", accuracy_score(y_test, pred_leak))

# ✅ CORRECT Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)
print("Correct Accuracy:", accuracy_score(y_test, pred))
