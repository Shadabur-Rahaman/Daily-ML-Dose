# Transformer Basics: Attention Calculation Demo

import torch
import torch.nn.functional as F

# Dummy input
Q = torch.tensor([[1.0, 0.0]])
K = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
V = torch.tensor([[10.0, 0.0], [0.0, 10.0]])

# Scaled Dot Product Attention
scores = Q @ K.T / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
weights = F.softmax(scores, dim=-1)
output = weights @ V

print("Attention Weights:", weights)
print("Output:", output)
