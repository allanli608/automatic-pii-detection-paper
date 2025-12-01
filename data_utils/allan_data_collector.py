import os
import zipfile
from dotenv import load_dotenv

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi


def download_data():
    COMPETITION_NAME = "pii-detection-removal-from-educational-data"
    DATA_PATH = "./data"
    ZIP_PATH = f"{DATA_PATH}/{COMPETITION_NAME}.zip"

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {COMPETITION_NAME}...")

    api.competition_download_files(COMPETITION_NAME, path=DATA_PATH)

    print("Download complete. Unzipping...")

    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_PATH)

        os.remove(ZIP_PATH)
        print("Unzipped and cleaned up.")
    else:
        print("Could not find the zip file to extract.")


if __name__ == "__main__":
    download_data()
