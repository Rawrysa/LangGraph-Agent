import logging

def setup_logger(name, log_file, level=logging.DEBUG):
    """Function to setup a logger with a file handler."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create a file handler
    file_handler = logging.FileHandler('logs/' + log_file)
    file_handler.setFormatter(formatter)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
