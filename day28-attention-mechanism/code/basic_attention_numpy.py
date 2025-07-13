import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    weights = softmax(scores)
    return np.matmul(weights, V), weights

# Toy example (batch=1, seq_len=4, d_k=64)
np.random.seed(42)
Q = np.random.rand(1, 4, 64)
K = np.random.rand(1, 4, 64)
V = np.random.rand(1, 4, 64)

output, weights = attention(Q, K, V)
print("Attention Output Shape:", output.shape)
print("Attention Weights Shape:", weights.shape)
