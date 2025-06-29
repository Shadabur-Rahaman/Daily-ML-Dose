# loss_functions_demo.py

import numpy as np
from sklearn.metrics import mean_squared_error, log_loss
import torch
import torch.nn as nn

# --------- MSE Demo (Regression) ----------
y_true_reg = np.array([3.0, -0.5, 2.0, 7.0])
y_pred_reg = np.array([2.5, 0.0, 2.1, 7.8])

mse = mean_squared_error(y_true_reg, y_pred_reg)
print("Mean Squared Error (MSE):", mse)

# --------- BCE Demo (Binary Classification) ----------
y_true_bin = np.array([1, 0, 1, 0])
y_pred_bin = np.array([0.9, 0.1, 0.8, 0.3])

bce = log_loss(y_true_bin, y_pred_bin)
print("Binary Cross Entropy (BCE):", bce)

# --------- Cross Entropy (Multi-Class) ----------
y_true_class = torch.tensor([2])  # class index
y_pred_logits = torch.tensor([[1.0, 2.0, 3.0]])  # logits for 3 classes

ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(y_pred_logits, y_true_class)

print("Cross-Entropy Loss:", loss.item())
