import zipfile
import os
import cv2
import numpy as np
import shutil
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
# Step 1: Extracting the zip file
zip_path = "pollen grains dataset.zip"
extract_path = "pollen_dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ Dataset extracted!")

# Step 2: Organizing the dataset into subfolders
for filename in os.listdir(extract_path):
    if filename.endswith(".jpg"):
        class_name = filename.split("_")[0].split("(")[0].strip().lower()
        class_folder = os.path.join(extract_path, class_name)

        if os.path.exists(class_folder) and not os.path.isdir(class_folder):
            print(f"❌ Skipping: '{class_folder}' exists as a file.")
            continue

        os.makedirs(class_folder, exist_ok=True)

        try:
            shutil.move(os.path.join(extract_path, filename),
                        os.path.join(class_folder, filename))
        except Exception as e:
            print(f"⚠️ Could not move '{filename}': {e}")

# Step 3: Loading images
IMAGE_SIZE = (64, 64)

def load_dataset(path):
    X, y = [], []
    print("📥 Loading images...")
    for label in os.listdir(path):
        label_folder = os.path.join(path, label)
        if not os.path.isdir(label_folder):
            continue
        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Skipped: {img_path}")
                continue
            img = cv2.resize(img, IMAGE_SIZE).flatten()
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y)

# Step 4: Training and evaluate model
X, y = load_dataset(extract_path)

if len(X) < 2:
    print("🚫 Not enough data")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42)
    
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Visualize sample predictions
    for i in range(min(5, len(X_test))):
        plt.imshow(X_test[i].reshape(IMAGE_SIZE), cmap='gray')
        plt.title(f"Predicted: {y_pred[i]}")
        plt.axis('off')
        plt.show()
