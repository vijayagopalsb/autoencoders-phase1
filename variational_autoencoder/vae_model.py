# File: variational_autoencoder/vae_model.py

# Import Libraries

# Add this BEFORE importing tensorflow
import os
import sys

# 3 = suppress all messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

class VAE(Model):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(latent_dim * 2)  # Outputs μ and log(σ²)
        ])

        # Decoder (same as vanilla AE)
        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
        ])

    def reparameterize(self, mu, log_var):
        batch_size = tf.shape(mu)[0]  # Dynamically get batch size
        eps = tf.random.normal(shape=(batch_size, self.latent_dim))
        return mu + tf.exp(log_var * 0.5) * eps

    def call(self, x):
        # Encode
        z_params = self.encoder(x)
        mu, log_var = tf.split(z_params, 2, axis=1)
        z = self.reparameterize(mu, log_var)
        # Decode
        reconstructed = self.decoder(z)
        return reconstructed, mu, log_var