import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

# This script performs texture-based image classification using a simple rule-based approach.
# It extracts texture features from images using the Gray-Level Co-occurrence Matrix (GLCM),
# then classifies images using threshold-based decision rules.

# Function to extract GLCM features from a grayscale image
def extract_glcm_features(gray_img, distances=[1], angles=[0], levels=256):
    gray_img = gray_img.astype(np.uint8)
    glcm = graycomatrix(gray_img, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm], dtype=np.float32)

# Function to extract GLCM features from a set of images
def images_to_glcm_features(X_4d):
    feature_list = []
    for i in range(X_4d.shape[0]):
        img = X_4d[i]
        if img.ndim == 3 and img.shape[-1] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        feats = extract_glcm_features(gray)
        feature_list.append(feats)
    return np.array(feature_list)

# Function to classify images using a simple rule-based approach
def classify_images(X_test, threshold=0.5):
    X_test_feat = images_to_glcm_features(X_test)
    predictions = []
    for feats in X_test_feat:
        if feats[0] > threshold:  # Classify based on contrast threshold
            predictions.append(1)
        else:
            predictions.append(0)
    return np.array(predictions)

# Function to evaluate the rule-based classifier
def evaluate_rule_based(X_test, Y_test):
    Y_test_1d = np.argmax(Y_test, axis=1)
    preds = classify_images(X_test)
    accuracy = np.mean(preds == Y_test_1d)
    print(f"Rule-based Classification - Accuracy: {accuracy}")
