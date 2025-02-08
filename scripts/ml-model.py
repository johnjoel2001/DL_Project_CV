import numpy as np
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# This script performs texture-based image classification using an SVM classifier.
# It extracts texture features from images using the Gray-Level Co-occurrence Matrix (GLCM),
# then trains an SVM model to classify the breast cancer images based on these features.
# Performance is evaluated using accuracy, precision, recall, and AUC.

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

# Function to train and evaluate an SVM classifier
def train_and_evaluate_svm(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    Y_train_1d = np.argmax(Y_train, axis=1)
    Y_val_1d = np.argmax(Y_val, axis=1)
    Y_test_1d = np.argmax(Y_test, axis=1)
    
    X_train_feat = images_to_glcm_features(X_train)
    X_val_feat = images_to_glcm_features(X_val)
    X_test_feat = images_to_glcm_features(X_test)
    
    clf = svm.SVC(kernel='linear', probability=True, random_state=42)
    clf.fit(X_train_feat, Y_train_1d)
    
    val_preds = clf.predict(X_val_feat)
    val_acc = accuracy_score(Y_val_1d, val_preds)
    val_prec = precision_score(Y_val_1d, val_preds, average='weighted')
    val_rec = recall_score(Y_val_1d, val_preds, average='weighted')
    val_auc = roc_auc_score(Y_val_1d, clf.predict_proba(X_val_feat), multi_class='ovr')
    print(f"Validation - Accuracy: {val_acc}, Precision: {val_prec}, Recall: {val_rec}, AUC: {val_auc}")
    
    test_preds = clf.predict(X_test_feat)
    test_acc = accuracy_score(Y_test_1d, test_preds)
    test_prec = precision_score(Y_test_1d, test_preds, average='weighted')
    test_rec = recall_score(Y_test_1d, test_preds, average='weighted')
    test_auc = roc_auc_score(Y_test_1d, clf.predict_proba(X_test_feat), multi_class='ovr')
    print(f"Test - Accuracy: {test_acc}, Precision: {test_prec}, Recall: {test_rec}, AUC: {test_auc}")
    
    joblib.dump(clf, "svm_glcm_model.joblib")
    print("Model saved to svm_glcm_model.joblib")
