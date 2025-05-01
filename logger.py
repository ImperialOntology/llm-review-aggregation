"""
Configures a logger that outputs to both the terminal and a file.
"""
import logging


level = logging.INFO  # Set the default logging level to INFO
log_file = 'run.log'

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(level)

# Create a formatter to define the log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a handler for terminal output (StreamHandler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Create a handler for file output (FileHandler)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
