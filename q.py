import streamlit as st
from PIL import Image
import numpy as np

import tensorflow as tf
model = tf.keras.models.load_model("model.keras")

st.title("Emotion Detection Using CNN ")
st.write("Capture a photo to predict your emotion.")

camera = st.camera_input("Take a picture")

if camera is not None:

    img = Image.open(camera).convert("RGB")

    img_resized = img.resize((100, 100))

    img_array = np.array(img_resized).astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    out = np.argmax(prediction)
    if out == 0:
        st.success("Angry") 
    elif out == 1:
        st.success("Fear") 
    elif out == 2:
        st.success("Happy") 
    elif out == 3:
        st.success("Sad") 
    elif out == 4:
        st.success("Suprise")
    else:
        pass

