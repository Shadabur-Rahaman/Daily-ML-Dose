# 🧰 Day 41 – Regularization Techniques  
**#DailyMLDose** | Controlling Overfitting in Machine Learning

Regularization is the secret sauce that keeps machine learning models from memorizing the training data and failing on new data. It introduces **penalty terms** or **constraints** to prevent overfitting and improve generalization.

---

## 🔍 Overview  
Today we cover:

- 📉 L1 and L2 Regularization
- 🔥 Dropout in Neural Networks
- 🎯 Early Stopping
- 🧠 Batch Normalization (acts as regularizer)
- 🔄 Data Augmentation
- 📈 Visualization of Effects

---

## 🖼️ Visuals

### 1. L1 vs L2 Penalty Intuition  
<img src="images/l1_vs_l2_penalty.png" width="600"/>

---

### 2. Dropout: Random Neuron Deactivation  
<img src="images/dropout_illustration.png" width="600"/>

---

### 3. Training Curves: Early Stopping  
<img src="images/early_stopping_curves.png" width="600"/>

---

### 4. BatchNorm as Regularizer  
<img src="images/batchnorm_regularization.png" width="600"/>

---

## 🧪 Code Highlights

### ✅ L2 Regularization with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)  # L2 regularization
```
✅ L1 Regularization

```python

l1_lambda = 0.001
l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())
loss = criterion(output, target) + l1_lambda * l1_penalty
```
✅ Dropout in a Neural Network
```python

import torch.nn.functional as F

class DropoutNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```
✅ Early Stopping Example

```python

best_loss = float('inf')
patience = 3
counter = 0

for epoch in range(100):
    train(...)
    val_loss = validate(...)
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break
```

📁 Folder Structure
```css

📁 day41-regularization-techniques/
├── code/
│   ├── l1_regularization.py
│   ├── l2_regularization.py
│   ├── dropout_demo.py
│   ├── early_stopping_example.py
│   └── batchnorm_regularizer.py
│
├── images/
│   ├── l1_vs_l2_penalty.png
│   ├── dropout_illustration.png
│   ├── early_stopping_curves.png
│   └── batchnorm_regularization.png
└── README.md
```

📚 References
Deep Learning Book – Ian Goodfellow, Chapter 7

CS231n Regularization

PyTorch Docs: Dropout

🔁 Navigation
⬅️ 
➡️ Day 42 – Coming Soon

🔗 Related Posts
![Day 40 – Attention Mechanisms](https://github.com/Shadabur-Rahaman/Daily-ML-Dose/tree/main/day40-attention-mechanisms)

🙌 Let’s Connect!
📎 Connect With Me
- 🔗 [Follow Shadabur Rahaman on LinkedIn](https://www.linkedin.com/in/shadabur-rahaman-1b5703249)
---
⭐ Star the GitHub Repo
---
🔁 Share this if it helped!
