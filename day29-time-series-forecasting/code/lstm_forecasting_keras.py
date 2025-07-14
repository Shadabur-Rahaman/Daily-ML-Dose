import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("your_timeseries.csv", parse_dates=['Date'], index_col='Date')
data = df['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 30
X, y = create_sequences(scaled_data, seq_len)

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(seq_len, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=16, verbose=1)

# Predict next 10 values
last_sequence = scaled_data[-seq_len:].reshape(1, seq_len, 1)
forecast = []
current_seq = last_sequence

for _ in range(10):
    next_val = model.predict(current_seq, verbose=0)[0]
    forecast.append(next_val)
    current_seq = np.append(current_seq[:, 1:, :], [[next_val]], axis=1)

forecast = scaler.inverse_transform(forecast)

# Plot
plt.plot(df.index[-100:], data[-100:], label='Historical')
future_index = pd.date_range(df.index[-1], periods=11, freq='D')[1:]
plt.plot(future_index, forecast, label='Forecast', linestyle='--')
plt.title("LSTM Forecast")
plt.legend()
plt.show()
