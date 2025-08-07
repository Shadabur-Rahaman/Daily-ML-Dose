from sklearn.metrics import accuracy_score

def detect_concept_drift(y_hist, y_pred_hist, y_new, y_pred_new, threshold=0.1):
    old_acc = accuracy_score(y_hist, y_pred_hist)
    new_acc = accuracy_score(y_new, y_pred_new)
    if abs(new_acc - old_acc) > threshold:
        print(f"⚠️ Concept drift detected! Δacc={new_acc - old_acc:.2f}")
