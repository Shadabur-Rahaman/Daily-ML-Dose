# 📈 Day 7 – ROC vs Precision-Recall (PR) Curves in ML

Welcome to **Day 7** of #DailyMLDose!

Today, we compare two of the most important evaluation tools for classifiers:  
**ROC Curve** and **Precision-Recall (PR) Curve**.

---

## 🚦 What They Are

### ✅ ROC Curve (Receiver Operating Characteristic)
- Plots **True Positive Rate (Recall)** vs **False Positive Rate**
- X-axis: FPR = FP / (FP + TN)  
- Y-axis: TPR = TP / (TP + FN)

### 🎯 Precision-Recall Curve
- Plots **Precision** vs **Recall**
- Useful when **positive class is rare** (imbalanced datasets)

---
🗂️ Folder Structure – day07-roc-vs-pr-curves/
```
day07-roc-vs-pr-curves/
├── README.md
├── roc_vs_pr_curves.py             # Python script
├── roc_curve_plot.png              # ROC curve visual
├── pr_curve_plot.png               # PR curve visual
└── roc_vs_pr_diff.png              # Summary comparison graphic
---
```
## ⚖️ ROC vs PR – Key Differences
```
| Feature                     | ROC Curve                  | PR Curve                        |
|----------------------------|----------------------------|---------------------------------|
| Focus                      | TPR vs FPR                 | Precision vs Recall             |
| Good for                   | Balanced datasets          | Imbalanced datasets             |
| Area Under Curve           | AUC-ROC                    | AUC-PR                          |
| More sensitive to          | TN values (FPR)            | FP/TP imbalance                 |
```
---

## 🧪 Python Demo

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)

# Train model
model = LogisticRegression()
model.fit(X, y)
y_scores = model.predict_proba(X)[:, 1]

# ROC
fpr, tpr, _ = roc_curve(y, y_scores)
roc_auc = auc(fpr, tpr)

# PR
precision, recall, _ = precision_recall_curve(y, y_scores)
pr_auc = auc(recall, precision)

# Plot both
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"AUC-ROC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f"AUC-PR = {pr_auc:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.tight_layout()
plt.show()
```
📊 Visual Summary
ROC Curve: Better for balanced datasets 

PR Curve: More informative for imbalanced data (rare positive class)

📉 Example:
In fraud detection or disease diagnosis, where positives are rare, PR curves are preferred.

🔁 Previous Post:
Day 6 → Confusion Matrix

🧠 Visual & Code Credits:
ROC vs PR intuition: @alec_helbling

ROC vs PR comparison chart: @chrisalbon

Curve demo via scikit-learn

📌 Follow Shadabur Rahaman

⭐ Star the GitHub repo

 [Follow Shadabur Rahaman on LinkedIn](https://www.linkedin.com/in/shadabur-rahaman-1b5703249/)  
