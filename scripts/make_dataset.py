"""
make_dataset.py downloads and extracts BreakHis dataset from the Kaggle API into the 'data/raw' folder.  

To run this script, we must provide our Kaggle user ID and API token, which can be obtained from:  
https://www.kaggle.com/settings/account  

Ensure to put in the Kaggle credentials on the CLI, when asked.
"""

import pandas as pd
import os
import opendatasets as od

def download_kaggle_dataset(dataset_url: str, target_dir: str) -> None:
    """
    Downloads a Kaggle dataset to a specified directory.

    Args:
        dataset_url (str): The URL of the Kaggle dataset.
        target_dir (str): The local directory where the dataset should be downloaded.

    Returns:
        None
    """
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Download the dataset to the specified folder
    od.download(dataset_url, data_dir=target_dir)

# Define the Kaggle dataset URL
dataset_url = 'https://www.kaggle.com/datasets/ambarish/breakhis'

# Define the target directory for storing the dataset
target_directory = 'data/raw'

# Call the function to download the dataset
download_kaggle_dataset(dataset_url, target_directory)
