import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from torchvision import transforms

def create_directories():
    """Creates necessary directories for data processing."""
    directories = [AUGMENTED_DIR, BENIGN_DIR, MALIGNANT_DIR, PROCESSED_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def collect_files(directory):
    """Recursively retrieves all PNG files in a given directory."""
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith(".png")]

def copy_images(files, destination):
    """Copies image files to a specified directory."""
    for file in files:
        shutil.copy(file, destination)

def balance_dataset(data):
    """Balances the dataset by upsampling the minority class."""
    benign_upsampled = resample(data[data['target'] == 0], n_samples=data[data['target'] == 1].shape[0], random_state=42)
    return pd.concat([data[data['target'] == 1], benign_upsampled])

def preprocess_images(image_paths):
    """Loads and normalizes images."""
    images = [tf.keras.utils.img_to_array(tf.keras.utils.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))) / 255.0 for img_path in tqdm(image_paths)]
    return np.array(images)

def split_data(X, y):
    """Performs K-Fold cross-validation split."""
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        for val_idx, test_idx in kf.split(X_test, y_test):
            X_val, X_test_final = X_test[val_idx], X_test[test_idx]
            y_val, y_test_final = y_test[val_idx], y_test[test_idx]
            return X_train, X_val, X_test_final, y_train, y_val, y_test_final

def augment_data(X_train, X_val, X_test_final):
    """Applies data augmentation."""
    data_gen = ImageDataGenerator(
        rotation_range=40 if len(benign_images) < len(malignant_images) else 25,
        zoom_range=0.25 if len(benign_images) < len(malignant_images) else 0.15,
        width_shift_range=0.3 if len(benign_images) < len(malignant_images) else 0.2,
        height_shift_range=0.3 if len(benign_images) < len(malignant_images) else 0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2]
    )
    data_gen.fit(X_train)
    data_gen.fit(X_val)
    data_gen.fit(X_test_final)
    return data_gen

def save_datasets(X_train, X_val, X_test_final, Y_train, Y_val, Y_test):
    """Saves processed datasets as NumPy arrays."""
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test_final)
    np.save(os.path.join(PROCESSED_DIR, "Y_train.npy"), Y_train)
    np.save(os.path.join(PROCESSED_DIR, "Y_val.npy"), Y_val)
    np.save(os.path.join(PROCESSED_DIR, "Y_test.npy"), Y_test)

# Define constants
IMAGE_SIZE = 128
AUGMENTED_DIR = "augmented"
BENIGN_DIR = os.path.join(AUGMENTED_DIR, "benign")
MALIGNANT_DIR = os.path.join(AUGMENTED_DIR, "malignant")
PROCESSED_DIR = "data/processed"

# Execute processing pipeline
create_directories()
benign_files = collect_files('data/raw/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/benign')
malignant_files = collect_files('data/raw/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/malignant')
copy_images(benign_files, BENIGN_DIR)
copy_images(malignant_files, MALIGNANT_DIR)
benign_images = collect_files(BENIGN_DIR)
malignant_images = collect_files(MALIGNANT_DIR)

data = pd.DataFrame({'image': benign_images + malignant_images, 'target': [0] * len(benign_images) + [1] * len(malignant_images)})
data_balanced = balance_dataset(data)

X = preprocess_images(data_balanced['image'])
y = data_balanced['target'].values
X_train, X_val, X_test_final, y_train, y_val, y_test_final = split_data(X, y)
Y_train, Y_val, Y_test = to_categorical(y_train, num_classes=2), to_categorical(y_val, num_classes=2), to_categorical(y_test_final, num_classes=2)

data_gen = augment_data(X_train, X_val, X_test_final)
save_datasets(X_train, X_val, X_test_final, Y_train, Y_val, Y_test)

print("Dataset processing complete.")
