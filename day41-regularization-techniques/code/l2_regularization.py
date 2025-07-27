import torch
import torch.nn as nn

model = nn.Linear(10, 1)
l2_loss = 0
for param in model.parameters():
    l2_loss += torch.sum(param ** 2)

print("L2 Regularization:", l2_loss.item())
