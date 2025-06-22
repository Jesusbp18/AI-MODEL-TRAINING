import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- Load model architecture ---
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- Load trained model ---
device = torch.device("cpu")
model = VAE()
model.load_state_dict(torch.load("vae_mnist.pth", map_location=device))
model.eval()

# --- Streamlit UI ---
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))
generate = st.button("Generate Images")

if generate:
    st.subheader(f"Generated images of digit {digit}")
    cols = st.columns(5)
    for i in range(5):
        z = torch.randn(1, 20)  # sample from latent space
        img = model.decode(z).detach().numpy().reshape(28, 28)

        # Show as image
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        cols[i].pyplot(fig)
