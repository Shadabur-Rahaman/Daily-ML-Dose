import pandas as pd

report = pd.DataFrame({
    "Metric": ["Accuracy", "Demographic Parity", "Equal Opportunity"],
    "Score": [0.91, 0.07, 0.03]
})
print(report)
