import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

# =============================
# 🔧 Step 1: Config & Setup
# =============================

BATCH_SIZE = 32
NUM_CLASSES = 10  # Update this based on your dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# 🧼 Step 2: Transforms
# =============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])

# =============================
# 📂 Step 3: Load Dataset
# =============================

# Replace this path with your actual dataset folder
train_dataset = datasets.FakeData(transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# =============================
# 🔁 Step 4: Load Pretrained Model
# =============================

model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final FC layer for our custom classification task
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(DEVICE)

# =============================
# 🧪 Step 5: Train the Classifier Head
# =============================

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Simple 1-epoch training loop (expand as needed)
model.train()
for images, labels in train_loader:
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("✅ Training complete with Transfer Learning on ResNet-18.")
