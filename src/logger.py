import logging
import os
from datetime import datetime

# log file name
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

# folder path without file name
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# final log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# logging config
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info(" logging has started")

