import numpy as np
from scipy.stats import entropy

def kl_divergence(p, q):
    return entropy(p, q)

# Simulated example
train_dist = np.array([0.2, 0.3, 0.5])
new_dist = np.array([0.3, 0.4, 0.3])

kl_score = kl_divergence(train_dist, new_dist)
print("KL Divergence:", kl_score)
