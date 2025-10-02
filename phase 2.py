import os
import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder

# Initialize FaceNet
embedder = FaceNet()

print("[INFO] Processing dataset...")

imagePaths = []
labels = []

# Loop through dataset/train/
for person_name in os.listdir("dataset/train"):
    person_dir = os.path.join("dataset/train", person_name)
    if not os.path.isdir(person_dir):
        continue
    
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        imagePaths.append(img_path)
        labels.append(person_name)

print(f"[INFO] Found {len(imagePaths)} images for {len(set(labels))} classes")

# Extract embeddings
embeddings = []
for (img_path, label) in zip(imagePaths, labels):
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get embeddings
    faces = embedder.extract(img, threshold=0.95)
    if len(faces) > 0:
        embeddings.append(faces[0]["embedding"])

# Encode labels
le = LabelEncoder()
labels_enc = le.fit_transform(labels[:len(embeddings)])

# Build data dictionary
data = {"embeddings": embeddings, "labels": labels_enc}

# Save embeddings
os.makedirs("embeddings", exist_ok=True)
with open("embeddings/face_embeddings.pkl", "wb") as f:
    pickle.dump(data, f)

print("[INFO] Embeddings saved to embeddings/face_embeddings.pkl")
