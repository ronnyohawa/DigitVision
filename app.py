import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

# Load the model (make sure it's saved in .h5 format)
@st.cache_resource
def load_model():
    # Check if the model exists
    if not os.path.exists('handwritten_digits.h5'):
        st.error("Model not found! Please train and save the model first.")
        return None
    else:
        # Load the saved model
        model = tf.keras.models.load_model('handwritten_digits.h5')
        return model

# Load the model
model = load_model()

# Streamlit App
if model:
    st.title("Handwritten Digit Recognition")
    st.write("Upload a handwritten digit image to predict.")

    # Image Upload
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Convert the uploaded image to an OpenCV format
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Preprocess the image
        img = cv2.resize(img, (28, 28))  # Resize to 28x28
        img = np.invert(img)  # Invert pixel values (MNIST is black on white, your images may be the opposite)
        img = img / 255.0  # Normalize to 0-1
        img = img.reshape(1, 28, 28)  # Reshape to fit the model's input shape

        # Model Prediction
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)

        # Display the prediction
        st.write(f"The model predicts this is a: **{predicted_digit}**")

        # Show the image
        st.image(img[0], caption="Uploaded Image", use_container_width=True)  # Updated parameter

        # Display the prediction probabilities
        st.write("Prediction probabilities:")
        for i in range(10):
            st.write(f"Digit {i}: {prediction[0][i]*100:.2f}%")
