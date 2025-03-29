# File: vanilla_autoencoder/model.py

# Import Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warnings, 3=errors

import tensorflow as tf # type: ignore
from tensorflow.keras import layers, Model # type: ignore

class VanillaAutoencoder(Model):
    def __init__(self, latent_dim=32):
        super(VanillaAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(latent_dim, activation='relu')  # Latent space
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(784, activation='sigmoid'),     # MNIST: 28x28=784
            layers.Reshape((28, 28))
        ])

    def call(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed