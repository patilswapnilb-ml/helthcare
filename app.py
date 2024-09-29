import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np


import gdown

# Download model from Google Drive
url = "https://drive.google.com/file/d/124UhbC0-gsRuHGPi_6cVRCQwKO_yj0mZ/view?usp=sharing"
output = "your_model_file.h5"  # or .pkl, etc.
gdown.download(url, output, quiet=False)

from keras.models import load_model
model = load_model("your_model_file.h5")
# Load the trained model


# Define the class names based on your dataset
class_names = ['Bacterial leaf blight', 'Brown spot', 'Healthy', 'Leaf Blast', 'Leaf scald', 'Narrow Brown Spot']

# Streamlit app
st.title("Rice Leaf Disease Detection")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = load_img(uploaded_file, target_size=(299, 299))

    # Display the image at 50% of the original size
    st.markdown(
        """
        <style>
        img {
            max-width: 50%;
            height: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.image(image, caption='Uploaded Image', use_column_width=False)

    # Preprocess the image
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Make predictions
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class_name = class_names[predicted_class[0]]

    st.title(f"Detected Diseases: {predicted_class_name}")
