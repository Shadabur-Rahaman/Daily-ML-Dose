# train_test_code_example.py

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load sample data
data = load_iris()
X, y = data.data, data.target

print("✅ Dataset loaded: Iris")
print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

# 1️⃣ Simple Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n🔍 Train-Test Split Accuracy:", accuracy_score(y_test, y_pred))

# 2️⃣ K-Fold Cross-Validation (k=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_accuracies = []

print("\n📂 K-Fold Cross-Validation (k=5):")
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    kf_accuracies.append(acc)
    print(f"  Fold {fold}: Accuracy = {acc:.4f}")

print("  Avg Accuracy:", np.mean(kf_accuracies))

# 3️⃣ Stratified K-Fold (preserves class proportions)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_accuracies = []

print("\n⚖️ Stratified K-Fold Cross-Validation (k=5):")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    skf_accuracies.append(acc)
    print(f"  Fold {fold}: Accuracy = {acc:.4f}")

print("  Avg Stratified Accuracy:", np.mean(skf_accuracies))
