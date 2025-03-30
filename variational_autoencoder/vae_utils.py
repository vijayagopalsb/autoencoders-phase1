# File: variational_autoencoder/vae_utils.py

# Import Libraries
import numpy as np 
import matplotlib.pyplot as plt

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0=all, 1=info, 2=warnings, 3=errors
# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
def vae_loss(reconstruction_loss, mu, log_var):
    """Calculate VAE loss = reconstruction loss + KL divergence"""
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
    return reconstruction_loss + kl_loss

def plot_reconstructions(model, test_images, n=10):
    """Plot original vs reconstructed images with latent space info"""
    # Ensure proper shape
    test_images = test_images.reshape(-1, 28, 28, 1)

    # Get predictions
    reconstructions, mu, log_var = model.predict(test_images[:n], verbose=0)

    plt.figure(figsize=(20, 6))

    for i in range(n):
        # Original
        plt.subplot(3, n, i+1)
        plt.imshow(test_images[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstructed
        plt.subplot(3, n, i+n+1)
        plt.imshow(reconstructions[i].squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

        # Latent stats (using np.exp)
        plt.subplot(3, n, i+2*n+1)
        plt.bar(range(len(mu[i])), mu[i])
        plt.title(f"μ={mu[i][0]:.2f}, σ²={np.exp(log_var[i])[0]:.2f}")

    plt.tight_layout()
    plt.savefig('vae_reconstructions.png')
    plt.close()