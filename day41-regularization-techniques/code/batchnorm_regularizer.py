import torch
import torch.nn as nn

x = torch.randn(8, 16)
bn = nn.BatchNorm1d(16)
out = bn(x)
print("BatchNorm Output Shape:", out.shape)
