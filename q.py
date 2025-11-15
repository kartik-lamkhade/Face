import streamlit as st
import cv2
import numpy as np
import joblib

# Load CNN Model
model = joblib.load("CNN_11.pkl")

# Define labels (order must match your model's output)
labels = ['Surprise','Fear','Yuck','Happy','Sad','Angry','no']

st.title("Emotion Detection Using CNN (100x100 Input)")
st.write("Capture a photo to predict your emotion.")

camera = st.camera_input("Take a picture")

if camera is not None:

    # Convert picture to numpy
    img_bytes = camera.getvalue()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input
    img_resized = cv2.resize(img, (100, 100)).astype('float32') / 255.0

    # Add batch dimension
    img_resized = np.expand_dims(img_resized, axis=0)

    # Predict
    prediction = model.predict(img_resized)
    class_id = np.argmax(prediction)

    # Final emotion
    emotion = labels[class_id]

    st.subheader("Predicted Emotion:")
    st.write(f"### 😃 {emotion}")
