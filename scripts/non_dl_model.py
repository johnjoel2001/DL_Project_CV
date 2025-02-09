import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from process_dataset_non_dl import extract_lbp_features, load_data_lbp

# Define dataset CSV path
csv_path = "data/raw/breakhis/Folds.csv"
dataset = pd.read_csv(csv_path)



#  Filter dataset 
dataset = dataset[dataset["mag"] == 40]
dataset = dataset.rename(columns={"filename": "path"})
dataset["label"] = dataset["path"].apply(lambda x: x.split("/")[3])
dataset["class"] = dataset["label"].apply(lambda x: 0 if x == "benign" else 1)

#  Train-Val-Test Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=1)
for train_index, val_test_index in sss.split(dataset["path"], dataset["class"]):
    train_df, val_test_df = dataset.iloc[train_index], dataset.iloc[val_test_index]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.67, random_state=1)
for val_index, test_index in sss.split(val_test_df["path"], val_test_df["class"]):
    val_df, test_df = val_test_df.iloc[val_index], val_test_df.iloc[test_index]

print("Train Size:", train_df.shape, "Val Size:", val_df.shape, "Test Size:", test_df.shape)

# Load Preprocessed Data
X_train, y_train = load_data_lbp(train_df)
X_test, y_test = load_data_lbp(test_df)

# Train ML Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel="linear", probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LogisticRegression": LogisticRegression(),
}

results = {}
os.makedirs("models", exist_ok=True)  #  Ensure model directory exists

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    joblib.dump(model, f"models/{model_name}_breakhis.pkl")  # Save in models folder

#  Print Model Accuracies
print("\nModel Performance:")
for model, acc in results.items():
    print(f"{model}: {acc:.2f}")

