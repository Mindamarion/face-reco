import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
from keras_facenet import FaceNet

# --- Page Config ---
st.set_page_config(page_title="Face Detection & Recognition", layout="centered")
st.title("👤 Minda Face Detection & Recognition 😎")

# --- Load Models ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    st.error("Error loading Haar Cascade XML file!")

try:
    embedder = FaceNet()
except Exception as e:
    st.error(f"Failed to load FaceNet model: {e}")
    st.stop()

try:
    recognizer = pickle.load(open("embeddings/face_recognizer.pkl", "rb"))
    label_encoder = pickle.load(open("embeddings/label_encoder.pkl", "rb"))
except FileNotFoundError:
    st.error("Trained recognizer or label encoder not found. Run Phase 2 & Phase 3 first.")
    st.stop()

# --- Instructions ---
st.markdown("""
### Instructions
1. Upload an image **or take a photo with your webcam**.
2. Choose the rectangle color for detected faces.
3. Adjust **scaleFactor** and **minNeighbors** for detection sensitivity.
4. Click **Detect Faces & Recognize** to process the image.
""")

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Detection Settings")
scaleFactor = st.sidebar.slider("Scale Factor", 1.05, 2.0, 1.05, 0.05)
minNeighbors = st.sidebar.slider("Min Neighbors", 1, 10, 3, 1)
color = st.sidebar.color_picker("Pick Rectangle Color", "#00FF00")
rect_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

# --- Input ---
st.subheader("Choose Input")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("Or take a photo with your webcam")

# --- Image Processing Function ---
def process_image(image):
    if image is None:
        st.warning("❌ No image to process")
        return

    try:
        rgb_image = np.array(Image.fromarray(image).convert("RGB"))
    except Exception as e:
        st.error(f"Failed to convert image: {e}")
        return

    if rgb_image is None or rgb_image.size == 0:
        st.warning("❌ Image conversion failed or image is empty")
        return

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        st.warning("⚠️ No faces detected. Try adjusting settings or using a frontal face image.")
        st.image(gray, caption="Grayscale Image used for detection", channels="GRAY")
        return

    for (x, y, w, h) in faces:
        cv2.rectangle(rgb_image, (x, y), (x + w, y + h), rect_color, 2)
        
        face_crop = rgb_image[y:y+h, x:x+w]
        try:
            embeddings = embedder.extract(face_crop, threshold=0.90)
            if embeddings:
                embedding_vector = embeddings[0]["embedding"].reshape(1, -1)
                preds = recognizer.predict_proba(embedding_vector)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = label_encoder.classes_[j]
                cv2.putText(
                    rgb_image,
                    f"{name}: {proba:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    rect_color,
                    2
                )
        except Exception as e:
            st.warning(f"FaceNet embedding error: {e}")

    # Display and download
    st.image(rgb_image, caption="Detected & Recognized Faces", channels="RGB")
    result_filename = "processed_faces.jpg"
    try:
        cv2.imwrite(result_filename, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        with open(result_filename, "rb") as file:
            st.download_button(
                "💾 Download Image",
                data=file,
                file_name=result_filename,
                mime="image/jpeg"
            )
    except Exception as e:
        st.warning(f"Failed to save or provide download: {e}")

# --- Main Logic ---
image_to_process = None
if uploaded_file is not None:
    try:
        image_to_process = np.array(Image.open(uploaded_file).convert("RGB"))
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
elif camera_input is not None:
    try:
        image_to_process = np.array(Image.open(camera_input).convert("RGB"))
    except Exception as e:
        st.error(f"Failed to read camera input: {e}")

if image_to_process is not None:
    if st.button("Detect Faces & Recognize"):
        process_image(image_to_process)
else:
    st.info("Upload an image or take a photo to start face detection and recognition.")
