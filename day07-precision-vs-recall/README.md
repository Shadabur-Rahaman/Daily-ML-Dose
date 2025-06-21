# 🎯 Day 7 – Precision vs Recall Tradeoff in Machine Learning

Welcome to **Day 7** of #DailyMLDose!

Today we dive into the important tradeoff between **Precision** and **Recall** — two powerful metrics often at odds in classification problems.

---

## 📌 Definitions

- **Precision** = TP / (TP + FP)  
  → Of all predicted positives, how many are correct?

- **Recall** = TP / (TP + FN)  
  → Of all actual positives, how many were caught?

---
### 🗂️ Folder Structure – 
```
day07-precision-vs-recall/
├── README.md
├── precision_vs_recall_curve.py        # Python demo script
├── precision_recall_curve.png          # Matplotlib/seaborn visual
└── precision_vs_recall_tradeoff.png    # Concept illustration
---
## 🎯 The Tradeoff

Improving one often hurts the other:
- Increasing **Recall** may lower **Precision** (more false positives)
- Increasing **Precision** may lower **Recall** (more false negatives)

📊 You often tune thresholds (e.g., model.predict_proba > 0.7) to balance them.

---

## 🔍 When to Prioritize

| Scenario                                | Focus On    |
|-----------------------------------------|-------------|
| Fraud Detection                         | Recall      |
| Spam Detection                          | Precision   |
| Medical Diagnosis (e.g., cancer)        | Recall      |
| Search Engines (relevant results)       | Precision   |

---

## 🧪 Python Code – Precision-Recall Curve

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, weights=[0.85, 0.15], random_state=42)
model = LogisticRegression()
model.fit(X, y)

y_scores = model.predict_proba(X)[:, 1]
precision, recall, thresholds = precision_recall_curve(y, y_scores)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
```
🖼️ Output:

📎 Visual Concept

A high threshold = High Precision, Low Recall

A low threshold = High Recall, Low Precision

🧠 Summary
Use Precision when false positives are costly

Use Recall when false negatives are dangerous

Use F1-Score when you need a balanced approach

🔁 Previous:
Day 6 → Confusion Matrix

🔔 Stay Updated

⭐ Star the GitHub repo

 [Follow Shadabur Rahaman on LinkedIn](https://www.linkedin.com/in/shadabur-rahaman-1b5703249/)  
