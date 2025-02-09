import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import lime
import lime.lime_image
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.segmentation import mark_boundaries
from process_dataset_non_dl import extract_lbp_features  # Import LBP feature extractor

# Load trained model
MODEL_PATH = "models/RandomForest_breakhis.pkl"  # Adjust path as needed
model = joblib.load(MODEL_PATH)

# Define dataset path
DATASET_PATH = "data/raw/breakhis/BreaKHis_v1/"
CSV_PATH = "data/raw/breakhis/Folds.csv"

# Load dataset CSV
dataset = pd.read_csv(CSV_PATH)

# Filter for test images (adjust as needed)
dataset = dataset[dataset["mag"] == 40]
dataset = dataset.rename(columns={"filename": "path"})
dataset["label"] = dataset["path"].apply(lambda x: x.split("/")[3])
dataset["class"] = dataset["label"].apply(lambda x: 0 if x == "benign" else 1)

# Function to preprocess LIME's input and predict using LBP features
def predict_with_lbp(image_batch):
    processed_features = []
    for img in image_batch:
        img_resized = resize(img, (128, 128))  # Resize for LBP
        features = extract_lbp_features(img_resized)
        processed_features.append(features)
    
    return model.predict_proba(np.array(processed_features))  # ✅ Returns probabilities

# Function to explain random test images using LIME
def explain_random_images_lime(test_df, num_samples=5):
    sample_test_images = test_df.sample(num_samples)  # ✅ Randomly select test images

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    for i, (_, row) in enumerate(sample_test_images.iterrows()):
        image_path = os.path.join(DATASET_PATH, row["path"])
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue  # Skip missing images

        image = imread(image_path)
        image_resized = resize(image, (128, 128))  # Resize for LBP
        features = extract_lbp_features(image_resized)

        # Convert features to NumPy array
        features_array = np.array(features).reshape(1, -1)

        # Get Prediction
        prediction = model.predict(features_array)[0]
        predicted_label = "Malignant (Yes)" if prediction == 1 else "Benign (No)"
        true_label = "Malignant (Yes)" if row["class"] == 1 else "Benign (No)"

        # Display Original Image
        axes[i, 0].imshow(image)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"True: {true_label}\nPredicted: {predicted_label}",
                             fontsize=10, color="red" if prediction != row["class"] else "green")

        # LIME Explanation
        explainer = lime.lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image_resized, predict_with_lbp, top_labels=2, hide_color=0, num_samples=1000
        )

        # Highlight **important regions** used for prediction
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True
        )
        axes[i, 1].imshow(mark_boundaries(temp, mask))
        axes[i, 1].axis("off")
        axes[i, 1].set_title("LIME: Important Regions")

        #  Highlight **positive & negative regions**
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False
        )
        axes[i, 2].imshow(mark_boundaries(temp, mask))
        axes[i, 2].axis("off")
        axes[i, 2].set_title("LIME: Positive & Negative Regions")

    plt.tight_layout()
    plt.show()

#  Run LIME Explanation
explain_random_images_lime(dataset, num_samples=5)
