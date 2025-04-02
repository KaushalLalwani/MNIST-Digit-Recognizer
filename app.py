import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model/mnist_digit_recognizer.h5")

# Streamlit UI
st.title("Handwritten Digit Recognizer")
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Preprocess image
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img)
    digit = np.argmax(prediction)

    st.image(img.reshape(28, 28), width=150, caption=f"Predicted Digit: {digit}")
