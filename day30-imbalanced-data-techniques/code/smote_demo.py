from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Generate imbalanced data
X, y = make_classification(
    n_samples=1000, n_features=2, n_redundant=0, 
    n_clusters_per_class=1, weights=[0.9, 0.1], random_state=42
)

print("Before SMOTE:", Counter(y))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("After SMOTE:", Counter(y_resampled))

# Visualize
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm')
plt.title("Before SMOTE")

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_resampled[:, 0], y=X_resampled[:, 1], hue=y_resampled, palette='coolwarm')
plt.title("After SMOTE")

plt.tight_layout()
plt.show()
