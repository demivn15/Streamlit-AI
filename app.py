import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Digit Image Classifier",
    page_icon="🔢",
    layout="centered"
)

# Load the pre-trained model with caching
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

model = load_model()

# App UI Header
st.title("🔢 Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9), and the pre-trained model will classify it.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    try:
        # Preprocessing to match the model's expected format (8x8 grayscale, flattened)
        img_gray = image.convert('L')
        img_resized = img_gray.resize((8, 8), Image.Resampling.LANCZOS)
        
        # Convert pixels to numpy array
        img_array = np.array(img_resized, dtype=float)
        
        # Scikit-learn digits dataset scales pixels from 0 to 16 (invert colors if background is white)
        # Note: adjust this if your uploaded image has a white background vs black background
        img_flattened = img_array.flatten()
        img_flattened = 16 - (img_flattened / 16.0) # Normalizing and mapping
        
        # Make prediction
        prediction = model.predict([img_flattened])
        prediction_proba = model.predict_proba([img_flattened])
        confidence = np.max(prediction_proba) * 100
        
        # Display Results
        st.success(f"### Prediction: Number {prediction[0]}")
        st.info(f"Confidence: {confidence:.2f}%")
            
    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
