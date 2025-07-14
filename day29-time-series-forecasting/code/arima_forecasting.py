import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load time series data (make sure Date is the index)
df = pd.read_csv("your_timeseries.csv", parse_dates=['Date'], index_col='Date')

# Visualize the data
df['Close'].plot(title="Original Time Series", figsize=(10, 4))
plt.show()

# Fit ARIMA model (p=5, d=1, q=0)
model = ARIMA(df['Close'], order=(5, 1, 0))
model_fit = model.fit()

# Summary and forecast
print(model_fit.summary())
forecast = model_fit.forecast(steps=10)
print("Next 10 Predictions:\n", forecast)

# Plot forecast
df['Close'].plot(label='Historical', figsize=(10, 4))
forecast.plot(label='Forecast', style='--')
plt.title("ARIMA Forecast")
plt.legend()
plt.show()
