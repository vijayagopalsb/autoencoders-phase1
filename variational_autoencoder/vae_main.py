# File: variational_autoencoders/main.py

import sys
import os

# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN optimizations

# Import Custom App Libraries
from logging_config import logger
from variational_autoencoder.vae_train import train_vae

# Begin here
if __name__ == "__main__":

    logger.info("Staring Variational Auto Encoding ...")
    train_vae()
    logger.info("Successfully Completed Variational Auto Encoding.")