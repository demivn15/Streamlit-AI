import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Image Classification App",
    page_icon="🖼️",
    layout="centered"
)

# Load the model with caching so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

model = load_model()

# App UI Header
st.title("🖼️ Image Classification App")
st.write("Upload an image, and the model will classify it.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Processing image...")
    
    try:
        # Preprocessing: Convert to grayscale, resize to match model input (8x8 for this example), and flatten
        img_gray = image.convert('L')
        img_resized = img_gray.resize((8, 8), Image.Resampling.LANCZOS)
        
        # Convert image pixels to numpy array and normalize (matching sklearn digits dataset scale)
        img_array = np.array(img_resized, dtype=float)
        img_flattened = img_array.flatten()
        
        # Rescale pixel values if your training data required it (e.g., scaling 0-255 to 0-16)
        img_flattened = img_flattened / 16.0 
        
        # Make prediction
        prediction = model.predict([img_flattened])
        prediction_proba = model.predict_proba([img_flattened]) if hasattr(model, "predict_proba") else None
        
        # Display Results
        st.success(f"### Prediction: Class {prediction[0]}")
        
        if prediction_proba is not None:
            confidence = np.max(prediction_proba) * 100
            st.info(f"Confidence: {confidence:.2f}%")
            
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
