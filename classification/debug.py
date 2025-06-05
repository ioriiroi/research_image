from pathlib import Path
import imghdr
import os
import sys
from PIL import Image


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setting import config

data_dir = config.DOWNLOAD_DIR
destination_dir = config.DOWNLOAD_DIR
image_extensions = [".png", ".jpg"]  # add there all your images file extensions

img_type_accepted_by_tf = ["jpeg", "png"]
for filepath in Path(data_dir).rglob("*"):
    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)
        if img_type is None:
            print(f"{filepath} is not an image")
        elif img_type not in img_type_accepted_by_tf:
            print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
            img = Image.open(filepath)
            destination_path = os.path.join(destination_dir, os.path.splitext(filepath)[0] + ".jpg")
            img.save(destination_path, 'JPG')
            img.close()
