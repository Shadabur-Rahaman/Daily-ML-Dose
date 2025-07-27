import torch
import torch.nn as nn

x = torch.randn(5, 5)
drop = nn.Dropout(p=0.3)
print("Before Dropout:\n", x)
print("After Dropout:\n", drop(x))
