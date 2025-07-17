# gan_discriminator_vs_generator.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Simple linear Generator and Discriminator (1D data for illustration)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, z):
        return self.gen(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

# Real data = Normal distribution
def real_data_sampler(n=1000):
    return torch.randn(n, 1) * 0.5 + 2

# Fake data = Generated from noise
def noise_sampler(n=1000):
    return torch.rand(n, 1) * 2 - 1

# Train dynamics visualized
def train_gan(steps=500):
    D = Discriminator()
    G = Generator()
    opt_D = torch.optim.Adam(D.parameters(), lr=0.001)
    opt_G = torch.optim.Adam(G.parameters(), lr=0.001)
    loss = nn.BCELoss()

    D_losses, G_losses = [], []

    for step in range(steps):
        # === Train D ===
        real_samples = real_data_sampler()
        fake_samples = G(noise_sampler()).detach()
        real_preds = D(real_samples)
        fake_preds = D(fake_samples)

        d_loss_real = loss(real_preds, torch.ones_like(real_preds))
        d_loss_fake = loss(fake_preds, torch.zeros_like(fake_preds))
        d_loss = (d_loss_real + d_loss_fake) / 2

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # === Train G ===
        fake_samples = G(noise_sampler())
        preds = D(fake_samples)
        g_loss = loss(preds, torch.ones_like(preds))

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        D_losses.append(d_loss.item())
        G_losses.append(g_loss.item())

        if (step + 1) % 100 == 0:
            print(f"Step {step+1} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # Plot
    plt.plot(D_losses, label="Discriminator Loss")
    plt.plot(G_losses, label="Generator Loss")
    plt.legend()
    plt.title("D vs G Loss Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    train_gan()
