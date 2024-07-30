import streamlit as st
import numpy as np
from PIL import Image
import pickle
# Load your pre-trained model
model = pickle.load(open(r"C:\Users\shubh\Desktop\Final\Framess\Framess.sav",'rb'))
# model.eval()

def preprocess_image(image):
    image = image.resize((256, 256))  # Resize image to 256,256,3
    image = np.array(image)  # Convert image to NumPy array
    image = image / 255.0  # Normalize to [0, 1] range
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

st.title("Framess: Real or Fake Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button('Predict'):
        try:
            # Preprocess the image
            input_image = preprocess_image(image)

            # Debug: Display shape of the input image
            st.write(f"Input image shape: {input_image.shape}")

            # Make prediction
            predictions = model.predict(input_image)
            confidence = np.max(predictions)
            predicted_class = np.argmax(predictions)

            # Map the predicted class to real or fake
            class_names = ['Real', 'Fake']  # Adjust based on your model's training classes
            prediction = class_names[predicted_class]

            st.write(f'Prediction: **{prediction}**')
        except ValueError as e:
            st.error(f"An error occurred: {e}")

if st.button('About'):
    st.write("This app uses a pre-trained deep learning model to predict if an uploaded image is real or fake.")
    st.write("Developed by Shubham Yadav")