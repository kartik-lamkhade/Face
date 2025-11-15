import streamlit as st
import cv2
import numpy as np
import joblib

# Load your saved CNN model
model = joblib.load("CNN_11.pkl")

# Same labels as your YOLO folder
labels = list(train_data.class_indices.keys()) if 'train_data' in globals() else ['Surprise','Fear','Yuck' ,'Happy','Sad' ,'Angry','no']

st.title("Emotion Detection Using CNN (300x300)")
st.write("Capture a photo to predict your emotion.")

camera = st.camera_input("Take a picture")

if camera is not None:

    # Convert camera input into numpy array
    img_bytes = camera.getvalue()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert BGR → RGB (because ImageDataGenerator gave RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to model input size
    img_resized = cv2.resize(img, (100,100))

    # Normalize exactly like training (rescale=1./255)
    img_resized = img_resized.astype('float32') / 255.0

    # Add batch dimension
    img_resized = np.expand_dims(img_resized, axis=0)

    # Predict emotion
    prediction = model.predict(img_resized)
   
    # Get highest probability label
    
    class_id = np.argmax(prediction)
    if class_id==1:
        emotion="Surprise"
    elif class_id==2:
        emotion="Fear"
    elif class_id==3:
        emotion="Yuck"
    elif class_id==4:
        emotion="Happy"
    elif class_id==5:
        emotion="Sad"
    elif class_id==6:
        emotion="Angry"
    elif class_id==7:
        emotion="no"
    emotion = labels[class_id]
    
    st.subheader("Predicted Emotion:")
    st.write(f"### 😃 {emotion}")
