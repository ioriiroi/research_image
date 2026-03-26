from dotenv import load_dotenv
load_dotenv()

import os

ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
REFRESH_TOKEN = os.getenv('REFRESH_TOKEN')
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR")
ILLUST_DIR = os.getenv("ILLUST_DIR")
DATA_JSON_DIR = os.getenv("DATA_JSON_DIR")
ILLUST_GOOD_DIR = os.getenv("ILLUST_GOOD_DIR")
ILLUST_FACE_DIR = os.getenv("ILLUST_FACE_DIR")
ILLUST_FACE_GOOD_DIR = os.getenv("ILLUST_FACE_GOOD_DIR")
DATA_DIR = os.getenv("DATA_DIR")
ILLUST_MIKU_DIR = os.getenv("ILLUST_MIKU_DIR")
MIKU_DATA_DIR = os.getenv("MIKU_DATA_DIR")