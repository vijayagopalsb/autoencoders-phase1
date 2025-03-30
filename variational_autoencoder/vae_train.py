# File: variational_autoencoder/vae_train.py

# Import Library

import os
import sys
import numpy as np
# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppresses ALL TensorFlow internal messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Import Custom App Library
from logging_config import logger
from vae_model import VAE
from vae_utils import vae_loss, plot_reconstructions

def train_vae(epochs=30, batch_size=128):

    # Data loading
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    logger.info(f"Input shape: {x_train.shape}")

    # Model and optimizer
    model = VAE(latent_dim=32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    @tf.function
    def train_step(x, kl_weight):
        with tf.GradientTape() as tape:
            reconstructions, mu, log_var = model(x)

            # Calculate losses
            rec_loss = tf.reduce_mean(
                tf.keras.losses.mse(
                    tf.reshape(x, [-1, 784]),
                    tf.reshape(reconstructions, [-1, 784])
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
            total_loss = rec_loss + kl_weight * kl_loss

        # Corrected variable name here (grads instead of gradients)
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return rec_loss, kl_loss

    # Training loop
    for epoch in range(epochs):
        kl_weight = min(1.0, (epoch+1)/epochs * 0.1)
        rec_losses, kl_losses = [], []

        # Training
        for batch in tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size):
            rec_loss, kl_loss = train_step(batch, kl_weight)
            rec_losses.append(rec_loss.numpy())
            kl_losses.append(kl_loss.numpy())

        # Validation
        val_rec, val_mu, val_log_var = [], [], []
        for val_batch in tf.data.Dataset.from_tensor_slices(x_test[:1000]).batch(batch_size):
            rec, mu, lv = model(val_batch)
            val_rec.append(rec.numpy())
            val_mu.append(mu.numpy())
            val_log_var.append(lv.numpy())

        val_rec = np.concatenate(val_rec).reshape(-1, 28, 28, 1)
        val_mu = np.concatenate(val_mu)
        val_log_var = np.concatenate(val_log_var)

        val_loss = np.mean((x_test[:1000] - val_rec)**2) + \
                  0.1 * np.mean(-0.5 * (1 + val_log_var - val_mu**2 - np.exp(val_log_var)))

        logger.info(f"Epoch {epoch+1}/{epochs}, "
              f"Rec Loss: {np.mean(rec_losses):.4f}, "
              f"KL Loss: {np.mean(kl_losses):.4f}, "
              f"Val Loss: {val_loss:.4f}")

    # Save and visualize
    model.save_weights('vae.weights.h5')
    plot_reconstructions(model, x_test[:10])