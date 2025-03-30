# File: variational_autoencoder/vae_model.py

# Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

class VAE(Model):
    def __init__(self, latent_dim=32):
        super().__init__()
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

        # Decoder (same as vanilla AE)
        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
        ])

        def reparameterize(self, mu, log_var):
            eps = tf.random.normal(shape=mu.shape)
            return mu + tf.exp(log_var * 0.5) * eps

        def call(self, x):
            # Encode
            z_params = self.encoder(x)
            mu, log_var = tf.split(z_params, 2, axis=1)
            z = self.reparameterize(mu, log_var)

            # Decode
            reconstructed = self.decoder(z)
            return reconstructed, mu, log_var

        def vae_loss(reconstruction_loss, mu, log_var):

            # Reconstruction loss (MSE)
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))

            return reconstruction_loss + kl_loss