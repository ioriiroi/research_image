from dotenv import load_dotenv
load_dotenv()

import os

ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
REFRESH_TOKEN = os.getenv('REFRESH_TOKEN')
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR")
ILLUST_DIR = os.getenv("ILLUST_DIR")