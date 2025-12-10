from pythonjsonlogger.json import JsonFormatter
import logging
import sys
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Stream out
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

# File out
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/sarcasm_classifier.log')
file_formatter = JsonFormatter(
    fmt='%(levelname)s %(name)s %(message)s %(asctime)s'
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)