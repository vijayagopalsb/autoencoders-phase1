# File: logging_cofig.py
# Step 1: Logging Configuration

# Define a centralized logger

import logging
import os
from pathlib import Path

def setup_logging():
    """Configure logging with safe directory creation."""
    try:
        # Define log directory path (using absolute path)
        log_dir = Path(__file__).parent / "log_file"
        log_file = log_dir / "autoencoder-phase1.log"

        # Create directory if needed (with exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s ->>> %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Logs will be saved to: {log_file}")
        return logger

    except Exception as e:
        # Fallback to console-only logging if file logging fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s ->>> %(message)s",
            handlers=[logging.StreamHandler()]
        )
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to initialize file logging: {e}. Using console logging only.")
        return logger

# Initialize logger
logger = setup_logging()