# Automated Feature Engineering with FeatureTools and tsfresh
import pandas as pd
import featuretools as ft

# Sample transactional data
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'user_id': [101, 101, 102, 103, 103],
    'amount': [250, 500, 450, 130, 670],
    'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='D')
})

# Create EntitySet
es = ft.EntitySet(id='transactions')
es = es.add_dataframe(dataframe_name='trans',
                      dataframe=df,
                      index='id',
                      time_index='timestamp')

# Automated Feature Engineering
feature_matrix, feature_defs = ft.dfs(entityset=es,
                                      target_dataframe_name='trans',
                                      agg_primitives=["sum", "mean", "mode"],
                                      trans_primitives=["month", "weekday"])

print("Generated Features:\n", feature_matrix.head())
