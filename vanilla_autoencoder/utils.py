# File: vanilla_autoencoder/utils.py

"""
WARNING FOR WSL USERS:
- Plot display (plt.show()) will not work in default WSL terminal
- All plots are automatically saved to 'reconstructions.png' instead
- To enable displays:
  1. Install X Server (VcXsrv/Xming)
  2. Run: export DISPLAY=$(grep -m 1 nameserver /etc/resolv.conf | awk '{print $2}'):0
"""

# Import Libraries
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warnings

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore

# Import App Libraries
# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from  logging_config import logger

def load_mnist():
    """Load MNIST dataset (returns train/test splits without labels)"""
    (x_train, _), (x_test, _) = mnist.load_data()
    return (x_train, _), (x_test, _)

def plot_results(model, test_images, n=10):

    """
    Save reconstruction comparisons (WSL-compatible)

    Args:
        model: Trained autoencoder model
        test_images: Test images to reconstruct
        n: Number of examples to show (default: 10)

    Note:
        For WSL users: This saves to 'reconstructions.png' instead of displaying
        due to terminal limitations. Check your working directory for the file.
    """
    reconstructions = model.predict(test_images[:n], verbose=0)

    plt.figure(figsize=(20, 4))

    for i in range(n):
        # Original image
        plt.subplot(2, n, i+1)
        plt.imshow(test_images[i], cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstructed image
        plt.subplot(2, n, i+n+1)
        plt.imshow(reconstructions[i], cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.savefig('reconstructions.png', bbox_inches='tight', dpi=300)
    plt.close()  # Prevents memory leaks
    logger.info("[WSL NOTE] Plot saved to 'reconstructions.png' (use an image viewer)")



