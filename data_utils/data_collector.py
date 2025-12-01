
import argparse
import os
import zipfile
import requests

def print_menu():
    menu = """
    Data Collector Menu:
    1. Download Default Dataset
    2. Download nbroad Dataset
    3. Download mpware Dataset
    4. Download All Datasets
    5. Exit
    """
    print(menu)


def download_dataset(url, folder_path):
    try:
        

        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, "dataset.zip")
        response = requests.get(url)
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Default Dataset downloaded successfully and saved to {save_path}")
        
        # Unzip the dataset
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        print(f"Dataset extracted to {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_default_dataset():
    url = "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/66653/7500999/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1764676302&Signature=Cuz31uGEow0hB6ipIJxHQ9SuDzxoUi7GVGBFNlsU0leDIA300xfGWTgTi8%2F6CAYtL7LPVap6c%2BZtEu1B5A6vA4ORV7xr11sPgvgCw%2F%2Bx3U20DqPJN2srpQLL%2FF74ArBXtCIImjqDo67aDbalOe0jApLZ7tdHHdDER%2BTzei6DA0Yqik6VK7vcAcZgOEW7g8K%2BQtMBgIiXrr1KY4Zuj0Tc9Pzv%2BBpbcJKwKBWfZbYb5rne6OCzzCLwAjZ%2BHMxE1GnwiQPZILlrG6BFUzLd5ZGeHRPEQVRpv%2BGOvV1sNW3ZFmCmw0CjAHXxAbCs%2FmlwMj38hgIvX1WT29ytXH0UBtFRtw%3D%3D&response-content-disposition=attachment%3B+filename%3Dpii-detection-removal-from-educational-data.zip"
    folder_path = "data/default_dataset"
    download_dataset(url, folder_path)

def download_nbroad_dataset():
    url = "https://www.kaggle.com/api/v1/datasets/download/nbroad/pii-dd-mistral-generated"
    folder_path = "data/nbroad_dataset"
    download_dataset(url, folder_path)

def download_mpware_dataset():
    url = "https://www.kaggle.com/api/v1/datasets/download/mpware/pii-mixtral8x7b-generated-essays"
    folder_path = "data/mpware_dataset"
    download_dataset(url, folder_path)

def download_all_datasets():
    download_default_dataset()
    download_nbroad_dataset()
    download_mpware_dataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PII data.")
    parser.add_argument("--download_all", action="store_true", help="Download all datasets")
    args = parser.parse_args()

    if args.download_all:
        download_all_datasets()
            
    else:
        while True:
            print_menu()
            choice = input("Enter your choice: ")
            
            if choice == '1':
                download_default_dataset()
            
            elif choice == '2':
                download_nbroad_dataset()

            elif choice == '3':
                download_mpware_dataset()
            elif choice == '4':
                download_all_datasets()
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")




    