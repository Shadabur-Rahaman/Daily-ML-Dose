# Datetime decomposition: Extract time-based features
import pandas as pd

# Sample datetime column
df = pd.DataFrame({
    'purchase_time': pd.to_datetime([
        '2023-01-01 08:45:00',
        '2023-06-15 12:30:00',
        '2023-09-10 19:15:00',
        '2023-11-05 21:00:00'
    ])
})

# Extracting components
df['hour'] = df['purchase_time'].dt.hour
df['day_of_week'] = df['purchase_time'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df['month'] = df['purchase_time'].dt.month
df['day'] = df['purchase_time'].dt.day

print("Datetime Features:\n", df)
