# phase3.py
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ===============================
# Load embeddings from Phase 2
# ===============================
print("[INFO] Loading embeddings...")
data = pickle.load(open("embeddings/face_embeddings.pkl", "rb"))
X, y = data['embeddings'], data['labels']

print(f"[INFO] Loaded {len(X)} embeddings for training/testing")

# ===============================
# Split train/test sets
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"[INFO] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ===============================
# Train SVM Classifier
# ===============================
print("[INFO] Training SVM classifier...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# ===============================
# Evaluate Accuracy
# ===============================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[INFO] Model accuracy: {acc * 100:.2f}%")

# ===============================
# Save the trained model
# ===============================
os.makedirs("embeddings", exist_ok=True)

# Save trained classifier
with open("embeddings/face_recognizer.pkl", "wb") as f:
    pickle.dump(model, f)

print("[INFO] Classifier saved to embeddings/face_recognizer.pkl")
