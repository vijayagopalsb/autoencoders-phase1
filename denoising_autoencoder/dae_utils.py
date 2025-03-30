# File: denoising_autoencoder/utils.py

# Import Libraries
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warnings

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore

# Import Custom App Libraries
from  logging_config import logger
# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def add_noise(images, noise_factor=0.5):
    """Add Guassian noise to images"""
    noisy = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy, 0.0, 1.0)  # Ensure pixel values stay in [0, 1]

def plot_denoising_results(model, clean_images, noise_factor=0.5, n=5):
    """Visualize original vs noisy vs reconstructed images"""
    noisy_images = add_noise(clean_images[:n], noise_factor)
    reconstructions = model.predict(noisy_images)

    plt.figure(figsize=(15, 6))
    for i in range(n):
        # Original
        plt.subplot(3, n, i+1)
        plt.imshow(clean_images[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Noisy
        plt.subplot(3, n, i+n+1)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        plt.title("Noisy")
        plt.axis('off')

        # Reconstructed
        plt.subplot(3, n, i+2*n+1)
        plt.imshow(reconstructions[i].squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.savefig('denoising_results.png')
    plt.close()