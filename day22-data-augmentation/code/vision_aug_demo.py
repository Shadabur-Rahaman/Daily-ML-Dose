from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load an image
img = Image.open("images/sample.jpg")  # Use your own image path

# Define augmentation pipeline
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

# Apply augmentation multiple times
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for i in range(4):
    aug_img = augment(img)
    axes[i].imshow(aug_img.permute(1, 2, 0))
    axes[i].axis('off')
    axes[i].set_title(f'Augmented #{i+1}')
plt.tight_layout()
plt.show()
