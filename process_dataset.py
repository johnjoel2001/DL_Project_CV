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

# Define constants
IMAGE_SIZE = 128
AUGMENTED_DIR = "augmented"
BENIGN_DIR = os.path.join(AUGMENTED_DIR, "benign")
MALIGNANT_DIR = os.path.join(AUGMENTED_DIR, "malignant")
PROCESSED_DIR = "data/processed"

# Create necessary directories
os.makedirs(BENIGN_DIR, exist_ok=True)
os.makedirs(MALIGNANT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def collect_files(directory):
    """Recursively retrieves all files in a given directory."""
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                file_list.append(os.path.join(root, file))
    return file_list

# Gather benign and malignant image paths
benign_files = collect_files('data/raw/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/benign')
malignant_files = collect_files('data/raw/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/malignant')


# Copy images to the augmented dataset directory
for file in benign_files:
    shutil.copy(file, BENIGN_DIR)
for file in malignant_files:
    shutil.copy(file, MALIGNANT_DIR)

# Retrieve image paths post-copy
benign_images = collect_files(BENIGN_DIR)
malignant_images = collect_files(MALIGNANT_DIR)

# Construct dataset DataFrame
image_paths = benign_images + malignant_images
targets = [0] * len(benign_images) + [1] * len(malignant_images)
data = pd.DataFrame({'image': image_paths, 'target': targets})

# Balance dataset by upsampling benign images
benign_upsampled = resample(data[data['target'] == 0],
                            n_samples=data[data['target'] == 1].shape[0],
                            random_state=42)
data_balanced = pd.concat([data[data['target'] == 1], benign_upsampled])
print(data_balanced['target'].value_counts())

# K-Fold Cross Validation
kf = KFold(n_splits=10, random_state=42, shuffle=True)

# Image transformation pipeline
data_transforms = transforms.Compose([
    transforms.Resize((220, 220)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.482, 0.455, 0.408], std=[0.231, 0.225, 0.223]),
])

# Load and preprocess images
train_images = []
for img_path in tqdm(data_balanced['image']):
    img = tf.keras.utils.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img = tf.keras.utils.img_to_array(img) / 255.0
    train_images.append(img)

X = np.array(train_images)
y = data_balanced['target'].values

# Split data using K-Fold
for train_idx, test_idx in kf.split(X, y):
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    for val_idx, test_idx in kf.split(X_test, y_test):
        X_val, X_test_final = X_test[val_idx], X_test[test_idx]
        y_val, y_test_final = y_test[val_idx], y_test[test_idx]

# Convert labels to categorical format
Y_train = to_categorical(y_train, num_classes=2)
Y_val = to_categorical(y_val, num_classes=2)
Y_test = to_categorical(y_test_final, num_classes=2)

# Print dataset shapes
print(X_train.shape, X_test_final.shape, X_val.shape)

# Data Augmentation
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

# Save dataset
np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val)
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test_final)
np.save(os.path.join(PROCESSED_DIR, "Y_train.npy"), Y_train)
np.save(os.path.join(PROCESSED_DIR, "Y_val.npy"), Y_val)
np.save(os.path.join(PROCESSED_DIR, "Y_test.npy"), Y_test)
