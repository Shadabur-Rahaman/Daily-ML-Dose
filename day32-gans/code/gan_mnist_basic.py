# gan_mnist_basic.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_dim=784):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),
            nn.Tanh()  # outputs in [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
lr = 0.0002
batch_size = 64
epochs = 50
noise_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Models
generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

# Loss & Optimizer
criterion = nn.BCELoss()
opt_gen = optim.Adam(generator.parameters(), lr=lr)
opt_disc = optim.Adam(discriminator.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    for idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.size(0)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake = generator(noise)
        disc_real = discriminator(real).view(-1)
        loss_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator(fake.detach()).view(-1)
        loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_real + loss_fake) / 2
        discriminator.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
        output = discriminator(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        generator.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {loss_disc:.4f} | G Loss: {loss_gen:.4f}")

    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            fake_images = generator(torch.randn(25, noise_dim).to(device)).reshape(-1, 1, 28, 28)
            grid = torchvision.utils.make_grid(fake_images, normalize=True)
            plt.imshow(np.transpose(grid.cpu(), (1, 2, 0)))
            plt.title(f"Epoch {epoch+1}")
            plt.axis('off')
            plt.show()
