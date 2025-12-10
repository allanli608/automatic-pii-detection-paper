import argparse
import os
import zipfile
from dotenv import load_dotenv

load_dotenv()


from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()


def print_menu():
    menu = """
    Data Collector Menu:
    1. Download Default Dataset (Competition)
    2. Download nbroad Dataset
    3. Download mpware Dataset
    4. Download All Datasets
    5. Exit
    """
    print(menu)

def unzip_file(extract_to):
    for item in os.listdir(extract_to):
        if not item.endswith(".zip"):
            continue
        file_path = os.path.join(extract_to, item)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(file_path)



def download_kaggle_competition(competition, folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Downloading {competition} dataset...")

        api.competition_download_files(
            competition,
            path=folder_path
        )

        unzip_file(folder_path)


        print(f"Downloaded and extracted to {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_kaggle_dataset(dataset, folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Downloading dataset {dataset}...")

        api.dataset_download_files(
            dataset,
            path=folder_path,
        )

        unzip_file(folder_path)

        print(f"Downloaded and extracted to {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_default_dataset():
    competition = "pii-detection-removal-from-educational-data"
    folder_path = "data/default_dataset"
    download_kaggle_competition(competition, folder_path)


def download_nbroad_dataset():
    dataset = "nbroad/pii-dd-mistral-generated"
    folder_path = "data/nbroad_dataset"
    download_kaggle_dataset(dataset, folder_path)


def download_mpware_dataset():
    dataset = "mpware/pii-mixtral8x7b-generated-essays"
    folder_path = "data/mpware_dataset"
    download_kaggle_dataset(dataset, folder_path)


def download_all_datasets():
    download_default_dataset()
    download_nbroad_dataset()
    download_mpware_dataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PII datasets.")
    parser.add_argument("--download_all", action="store_true")
    args = parser.parse_args()

    if args.download_all:
        download_all_datasets()
    else:
        while True:
            print_menu()
            choice = input("Enter your choice: ")

            if choice == "1":
                download_default_dataset()
            elif choice == "2":
                download_nbroad_dataset()
            elif choice == "3":
                download_mpware_dataset()
            elif choice == "4":
                download_all_datasets()
            elif choice == "5":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Try again.")
