# File: denoising_autoencoder/dae_model.py

# Import Libraries
import os
import sys
# Supress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors
# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf # type: ignore
from tensorflow.keras import layers, Model # type: ignore

#Import Custom App Libraries

from logging_config import logger

class DenoisingAutoEncoder(Model):

    def __init__(self, latent_dim=32):
        super().__init__()

        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(latent_dim, activation="relu")  # Bottleneck
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(784, activation="sigmoid"),
            layers.Reshape((28, 28, 1))
        ])

    def call(self, noisy_input):
        latent = self.encoder(noisy_input)
        reconstructed = self.decoder(latent)
        return reconstructed


