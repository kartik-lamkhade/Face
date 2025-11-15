import streamlit as st
from PIL import Image
import numpy as np
import joblib

# Load model
model = joblib.load("cnn_model.h5")

# Labels (same order as your model output)
labels = ['Surprise','Fear','Yuck','Happy','Sad','Angry','no']

st.title("Emotion Detection Using CNN ")
st.write("Capture a photo to predict your emotion.")

camera = st.camera_input("Take a picture")

if camera is not None:

    # Read image from camera
    img = Image.open(camera).convert("RGB")

    # Resize for model
    img_resized = img.resize((100, 100))

    # Convert to numpy + normalize
    img_array = np.array(img_resized).astype("float32") / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_id = np.argmax(prediction)

    emotion = labels[class_id]

    st.subheader("Predicted Emotion:")
    st.write(f"### 😃 {emotion}")
