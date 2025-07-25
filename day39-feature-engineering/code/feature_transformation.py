# Feature Transformation: Scaling, log-transform, power transforms
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
import numpy as np

# Sample data
df = pd.DataFrame({
    'income': [25000, 55000, 75000, 120000, 200000],
    'balance': [1500, 2500, 3200, 5000, 7600]
})

# Standard Scaling
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("Standard Scaled:\n", df_scaled)

# Log Transformation (avoid log(0) by adding a small constant)
df_log = np.log1p(df)
print("\nLog Transformed:\n", df_log)

# Power Transform
pt = PowerTransformer()
df_power = pd.DataFrame(pt.fit_transform(df), columns=df.columns)
print("\nPower Transformed (Yeo-Johnson):\n", df_power)
