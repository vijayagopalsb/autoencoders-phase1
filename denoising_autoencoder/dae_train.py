# File: denoising_autoencoder/dae_main.py

import os
import sys
# Supress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors
# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import tensorflow as tf

# Import Custom App Libraries
from denoising_autoencoder.dae_model import  DenoisingAutoEncoder
from denoising_autoencoder.dae_utils import add_noise, plot_denoising_results
from logging_config import logger

def train_dae(epochs=30, batch_size=256, noise_factor=0.5):
    # Load data
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Model and optimizer
    model = DenoisingAutoEncoder(latent_dim=32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size):
            noisy_batch = add_noise(batch, noise_factor)
            with tf.GradientTape() as tape:
                reconstructions = model(noisy_batch)
                loss = loss_fn(batch, reconstructions)  # Compare to CLEAN images
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss

        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(x_train)*batch_size:.4f}")

    # Save and visualize
    model.save_weights("dae.weights.h5")
    plot_denoising_results(model, x_test, noise_factor)