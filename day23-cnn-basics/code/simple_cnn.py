import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layer: input = 1 channel, output = 16 feature maps
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        # Pooling layer: downsample by 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer (optional)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Fully connected layer: 32 feature maps of 7x7 after 2 poolings from 28x28 input
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # 10 classes for classification

    def forward(self, x):
        # Conv + ReLU + Pool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the feature maps
        x = x.view(x.size(0), -1)  # batch_size x (32*7*7)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Test run
if __name__ == "__main__":
    model = SimpleCNN()
    print(model)

    dummy_input = torch.randn(1, 1, 28, 28)  # batch of 1 grayscale image (28x28)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: [1, 10]
