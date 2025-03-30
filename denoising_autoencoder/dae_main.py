# File: denoising_autoencoders/dae_main.py

import sys
import os

# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN optimizations

# Import Custom App Libraries
from logging_config import logger
from denoising_autoencoder.dae_train import train_dae

# Begin here
if __name__ == "__main__":

    logger.info("Staring Denoising Auto Encoding ...")
    train_dae()
    logger.info("Successfully Completed Denoising Auto Encoding.")