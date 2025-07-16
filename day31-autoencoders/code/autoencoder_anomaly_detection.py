import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Simulated data: Normal = Gaussian(0,1), Anomaly = Gaussian(5,1)
X_normal = np.random.normal(0, 1, (1000, 10))
X_anomaly = np.random.normal(5, 1, (100, 10))

# Build Autoencoder
model = Sequential([
    Dense(6, activation='relu', input_shape=(10,)),
    Dense(3, activation='relu'),
    Dense(6, activation='relu'),
    Dense(10, activation='linear')
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_normal, X_normal, epochs=50, batch_size=32, verbose=0)

# Evaluate on both sets
recon_normal = model.predict(X_normal)
recon_anomaly = model.predict(X_anomaly)

mse_normal = np.mean(np.power(X_normal - recon_normal, 2), axis=1)
mse_anomaly = np.mean(np.power(X_anomaly - recon_anomaly, 2), axis=1)

# Threshold
threshold = np.percentile(mse_normal, 95)

print("Anomalies Detected:", np.sum(mse_anomaly > threshold))

# Plot
plt.hist(mse_normal, bins=50, alpha=0.6, label='Normal')
plt.hist(mse_anomaly, bins=50, alpha=0.6, label='Anomaly')
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.title("Reconstruction Loss Distribution")
plt.show()
