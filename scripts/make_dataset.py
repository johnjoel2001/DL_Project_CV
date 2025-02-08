"""
make_dataset.py downloads and extracts the BreakHis dataset from the Kaggle API into the 'data/raw' folder.

To run this script, provide your Kaggle user ID and API token, which can be obtained from:
https://www.kaggle.com/settings/account

Ensure to input your Kaggle credentials on the CLI when prompted.
"""

import os
import opendatasets as od

def download_kaggle_dataset(dataset_url: str, target_dir: str) -> None:
    """
    Downloads a Kaggle dataset to a specified directory.

    Args:
        dataset_url (str): The URL of the Kaggle dataset.
        target_dir (str): The local directory where the dataset should be downloaded.
    """
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Download the dataset to the specified folder
    od.download(dataset_url, data_dir=target_dir)

if __name__ == "__main__":
    # Define the Kaggle dataset URL
    DATASET_URL = 'https://www.kaggle.com/datasets/ambarish/breakhis'
    
    # Define the target directory for storing the dataset
    TARGET_DIRECTORY = 'data/raw'
    
    # Call the function to download the dataset
    download_kaggle_dataset(DATASET_URL, TARGET_DIRECTORY)
    
    print(f"Dataset successfully downloaded to {TARGET_DIRECTORY}")

