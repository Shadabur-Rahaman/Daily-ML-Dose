import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample dataset (2 features)
X = np.array([
    [100, 1],
    [200, 2],
    [300, 3],
    [400, 4],
    [500, 5]
])

# Standardization: (x - mean) / std
scaler_standard = StandardScaler()
X_standardized = scaler_standard.fit_transform(X)

# Normalization: (x - min) / (max - min)
scaler_minmax = MinMaxScaler()
X_normalized = scaler_minmax.fit_transform(X)

print("Original:\n", X)
print("\nStandardized:\n", X_standardized)
print("\nNormalized:\n", X_normalized)
