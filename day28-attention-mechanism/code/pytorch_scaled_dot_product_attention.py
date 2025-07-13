import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights

# Example with batch size 2, sequence length 5, d_k = 64
torch.manual_seed(0)
Q = torch.rand(2, 5, 64)
K = torch.rand(2, 5, 64)
V = torch.rand(2, 5, 64)

output, weights = scaled_dot_product_attention(Q, K, V)
print("Output Shape:", output.shape)
print("Weights Shape:", weights.shape)
