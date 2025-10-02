# phase 4: real-time face recognition
import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet

# Load FaceNet model
embedder = FaceNet()

# Load classifier (trained in phase 3)
print("[INFO] Loading trained classifier...")
recognizer = pickle.load(open("embeddings/face_recognizer.pkl", "rb"))
label_encoder = pickle.load(open("embeddings/label_encoder.pkl", "rb"))

# Initialize webcam
print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face embeddings
    results = embedder.extract(rgb, threshold=0.95)

    for res in results:
        # Extract bounding box
        x1, y1, x2, y2 = res["box"]
        face_embedding = res["embedding"].reshape(1, -1)

        # Predict label
        preds = recognizer.predict_proba(face_embedding)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = label_encoder.classes_[j]

        # Draw box and label
        text = f"{name}: {proba:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Real-Time Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
