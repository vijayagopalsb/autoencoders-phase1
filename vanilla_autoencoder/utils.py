# File: vanilla_autoencoders/utils.py

# Import Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warnings, 3=errors

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore

def load_mnist():
    (x_train, _), (x_test, _) = mnist.load_data()
    return (x_train, _), (x_test, _)

def plot_results(model, test_images, n=10):
    """Plot original vs reconstructed images."""
    try:
        reconstructions = model.predict(test_images[:n])
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Original
            plt.subplot(2, n, i+1)
            plt.imshow(test_images[i], cmap='gray')
            plt.title("Original")
            plt.axis('off')

            # Reconstructed
            plt.subplot(2, n, i+n+1)
            plt.imshow(reconstructions[i], cmap='gray')
            plt.title("Reconstructed")
            plt.axis('off')
        plt.savefig('results.png')
        plt.show()
    except Exception as e:
        print(f"Error in plotting: {e}")
        raise



