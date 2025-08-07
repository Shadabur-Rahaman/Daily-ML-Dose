import logging
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)

def log_performance(y_true, y_pred):
    logging.info(f"Precision: {precision_score(y_true, y_pred):.2f}")
    logging.info(f"Recall:    {recall_score(y_true, y_pred):.2f}")
    logging.info(f"F1-score:  {f1_score(y_true, y_pred):.2f}")
