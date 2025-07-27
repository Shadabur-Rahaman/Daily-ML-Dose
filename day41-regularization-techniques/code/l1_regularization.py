from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=100, n_features=20, noise=10)

model_l1 = Lasso(alpha=0.1)
model_l2 = Ridge(alpha=1.0)

model_l1.fit(X, y)
model_l2.fit(X, y)

print("L1 Coefficients:", model_l1.coef_)
print("L2 Coefficients:", model_l2.coef_)
```
