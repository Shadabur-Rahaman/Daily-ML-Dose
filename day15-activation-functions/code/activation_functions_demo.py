# activation_functions_demo.py

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Define input range
x = np.linspace(-10, 10, 100)
x_tensor = torch.linspace(-10, 10, 100)

# Define activation functions
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)
gelu = F.gelu(x_tensor).numpy()

# Plot all
plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid, label='Sigmoid', color='blue')
plt.plot(x, tanh, label='Tanh', color='green')
plt.plot(x, relu, label='ReLU', color='orange')
plt.plot(x, gelu, label='GELU', color='purple')

plt.title("Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
plt.show()
