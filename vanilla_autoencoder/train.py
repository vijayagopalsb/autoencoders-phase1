# File: vanilla_autoencoder/train.py

# Import Libraries
import pickle
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

def model_and_image_reconstruction(): 

  
    # Configuration
    EPOCHS = 20
    BATCH_SIZE = 256
    LATENT_DIM = 32
    MODEL_NAME = 'vanilla_ae'

    logger.info("Starting MNIST Autoencoder Training")
    logger.info(f"Configuration: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, Latent Dim={LATENT_DIM}")

    # Load and prepare data

    (x_train, _), (x_test, _) = load_mnist()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    logger.info(f"Loaded MNIST data: {len(x_train)} training, {len(x_test)} test samples")
    # Model setup
    logger.info(f"Model setup: Latent Dimension as {LATENT_DIM}")
    autoencoder = VanillaAutoencoder(latent_dim=LATENT_DIM)
    autoencoder.compile(optimizer='adam', loss='mse')
    logger.info("Model initialized and compiled")

    # Custom callback for logging
    class LoggingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logger.info(
                f"Epoch {epoch+1}/{EPOCHS} - "
                f"loss: {logs['loss']:.4f} - "
                f"val_loss: {logs['val_loss']:.4f}"
            )

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
        validation_data=(x_test, x_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            LoggingCallback(),
            tf.keras.callbacks.ModelCheckpoint(
                'vanilla_ae.weights.h5',
                save_weights_only=True,
                monitor='val_loss',
                save_best_only=True
            )
        ],
        verbose=0  # Disable default progress bar
    )
    logger.info(f"Training completed. Final val_loss: {history.history['val_loss'][-1]:.4f}")

    # Save final weights (using new format)
    autoencoder.save_weights(f'{MODEL_NAME}.weights.h5')

    # Plot results
    plot_results(autoencoder, x_test)
    logger.info("Reconstructions saved to reconstructions.png")
    # Save training history
    with open(f'{MODEL_NAME}_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    logger.info("Training history saved to vanilla_ae_history.pkl")