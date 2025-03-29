# File: vanilla_autoencoder/train.py

# Import Libraries
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warnings
import tensorflow as tf # type: ignore
from model import VanillaAutoencoder
from utils import load_mnist, plot_results

# Import App Libraries
# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from  logging_config import logger

# Configuration
EPOCHS = 20
BATCH_SIZE = 256
LATENT_DIM = 32
MODEL_NAME = 'vanilla_ae'

# Load and prepare data
(x_train, _), (x_test, _) = load_mnist()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Model setup
autoencoder = VanillaAutoencoder(latent_dim=LATENT_DIM)
autoencoder.compile(optimizer='adam', loss='mse')

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        f'{MODEL_NAME}.weights.h5',
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
]

# Train
history = autoencoder.fit(
    x_train, x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=callbacks
)

# Save final weights (using new format)
autoencoder.save_weights(f'{MODEL_NAME}.weights.h5')

# Plot results
plot_results(autoencoder, x_test)

# Save training history
import pickle
with open(f'{MODEL_NAME}_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)