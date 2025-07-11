import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Toy input
Q = np.random.rand(1, 4, 64)  # (batch, seq_len, d_k)
K = np.random.rand(1, 4, 64)
V = np.random.rand(1, 4, 64)

# Scaled dot-product attention
d_k = Q.shape[-1]
scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
weights = softmax(scores)
output = np.matmul(weights, V)

print("Self-Attention Output Shape:", output.shape)
