import logging
import os

def get_logger(name: str) -> logging.Logger:
    """
    Creates a centralized logger that outputs to both the console and a file.
    """
    # 1. Ensure the logs directory exists at the root of the project
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "cortex.log")

    # 2. Initialize the logger
    logger = logging.getLogger(name)
    
    # Prevent duplicate logs if called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # 3. Define the professional formatting blueprint
        formatter = logging.Formatter(
            '%(asctime)s | %(name)-18s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 4. Console output (Terminal)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # 5. File output (Permanent Record)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # Attach handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger