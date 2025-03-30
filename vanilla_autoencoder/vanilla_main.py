# File: vanilla_autoencoders/main.py

import sys
import os

# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logging_config import logger
from vanilla_autoencoder.train import model_and_image_reconstruction 

if __name__ == "__main__":

    logger.info("Staring Vanilla Auto Encoding ...")
    model_and_image_reconstruction()
    logger.info("Successfully Completed Vanilla Auto Encoding.")