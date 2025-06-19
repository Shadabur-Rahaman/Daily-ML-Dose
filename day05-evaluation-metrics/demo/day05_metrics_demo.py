from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Sample ground truth and predicted labels
y_true = [1, 0, 1, 1, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1]

# Calculate basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Calculate specificity from confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)

# Display the results
print("📊 Evaluation Metrics:")
print(f"✅ Accuracy     : {accuracy:.2f}")
print(f"🎯 Precision    : {precision:.2f}")
print(f"♻️ Recall       : {recall:.2f}")
print(f"🛡️ Specificity  : {specificity:.2f}")
print(f"⚖️ F1 Score     : {f1:.2f}")
