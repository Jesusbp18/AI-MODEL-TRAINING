import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import requests

# --- Modelo VAE ---
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

# --- Descargar y cargar modelo ---
MODEL_URL = "https://drive.google.com/uc?export=download&id=1DYJ6ulF-q0fI8RO9-UDUXgfzW-jbbtPK"
MODEL_PATH = "vae_mnist.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "wb") as f:
            f.write(requests.get(MODEL_URL).content)
    device = torch.device("cpu")
    model = VAE()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()

# --- Interfaz Streamlit ---
st.title("Generador de Dígitos Manuscritos (MNIST)")
st.write("Selecciona un dígito (0–9) para generar imágenes sintéticas.")

digit = st.selectbox("Escoge un dígito:", list(range(10)))
generate = st.button("Generar imágenes")

if generate:
    st.subheader(f"Imágenes generadas del dígito {digit}")
    cols = st.columns(5)
    for i in range(5):
        z = torch.randn(1, 20)  # muestra aleatoria del espacio latente
        img = model.decode(z).detach().numpy().reshape(28, 28)

        # Mostrar imagen
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        cols[i].pyplot(fig)
