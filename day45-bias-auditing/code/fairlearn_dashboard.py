from fairlearn.widget import FairlearnDashboard

FairlearnDashboard(sensitive_features=gender, y_true=labels, y_pred=model_predictions)
