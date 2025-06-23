# 📊 Day 5 – Evaluation Metrics: Accuracy, Precision, Recall, F1, Specificity

Welcome to **Day 5** of #DailyMLDose!

Today’s concept is **Evaluation Metrics** — essential tools to judge how well your machine learning model performs.

---
## 🗂️ Folder Structure – day05-evaluation-metrics/
```
day05-evaluation-metrics/
|   └──demo/
|      └──  roc_vs_pr_curves.py     # Python script
├── confusion_matrix.png
├── metrics_formula.png
└──  README.md
---
```
## 📌 Key Metrics in Classification

### ✅ Accuracy
The ratio of correctly predicted observations to the total observations.
```
\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]
```
![Accuracy](Accuracy.jpg)

### 🎯 Precision
How many of the predicted positives are actually positive?
```
\[
\text{Precision} = \frac{TP}{TP + FP}
\]
```
![Precision](precision.jpg)

### ♻️ Recall (Sensitivity)
How many actual positives were correctly predicted?
```
\[
\text{Recall} = \frac{TP}{TP + FN}
\]
```
![Recall](recall.jpg)

### 🛡️ Specificity (True Negative Rate)
How many actual negatives were correctly predicted?
```
\[
\text{Specificity} = \frac{TN}{TN + FP}
\]
```
![Specificity](specificity.jpg)
### ⚖️ F1-Score
Harmonic mean of Precision and Recall. Good when you need a balance.
```
\[
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]
```
![Specificity](f1_score.png)
---
### Recall ✅ Accuracy 🎯 Precision ♻️ Recall (Sensitivity) ⚖️ F1-Score
![Accuracy, Precision, Recall, F1-Score](./Accuracy_Precision_Recall_f1.jpg)
---

## 📊 Confusion Matrix

A great way to visualize performance across all metrics:

![Confusion Matrix](Confusion_matrix.jpg)

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| **Actual Positive** | TP                 | FN                 |
| **Actual Negative** | FP                 | TN                 |

---

## 📍 When to Use What?

| Scenario                        | Best Metric        |
|---------------------------------|--------------------|
| Balanced classes                | Accuracy           |
| False positives are costly      | Precision          |
| False negatives are costly      | Recall             |
| You need a balance              | F1 Score           |

---

## 💡 Real-World Analogy

📬 **Spam Email Classification**

- **High Precision**: Very few non-spam emails marked as spam  
- **High Recall**: Most spam emails are correctly caught  
- **F1 Score**: Tradeoff between missing spam and flagging valid emails

---

## 🔗 Code Snippet (sklearn)

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

y_true = [1, 0, 1, 1, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1]

# Basic metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall (Sensitivity):", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))

# Specificity from confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

```
📎 Visual Aids

🔁 Previous Post:
Day 4 → [Train-Test Split](./day04-train-test-validation)

📌 Follow for more on [LinkedIn](https://www.linkedin.com/in/shadabur-rahaman-1b5703249/)
⭐ Star the GitHub repo
🎯 Let’s keep evaluating wisely!

