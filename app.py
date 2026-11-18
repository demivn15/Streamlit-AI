import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Classifier",
    page_icon="🔢",
    layout="centered"
)

# Load the pre-trained model with caching
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

model = load_model()

st.title("🔢 Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9). For best results, use a dark digit on a light background.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    try:
        # 1. Convert to grayscale ('L')
        img_gray = image.convert('L')
        
        # 2. Invert colors if background is white (sklearn digits expect dark background, bright foreground)
        # We check the corners to see if the background is light
        extrema = img_gray.getextrema()
        # Resize to 8x8 using high-quality resampling
        img_resized = img_gray.resize((8, 8), Image.Resampling.LANCZOS)
        
        img_array = np.array(img_resized, dtype=float)
        
        # Normalize and map pixels from 0-255 to sklearn's 0-16 scale
        # Automatically invert if it looks like dark text on a white background
        if np.mean(img_array) > 127: 
            img_array = 16 - (img_array / 255.0 * 16)
        else:
            img_array = (img_array / 255.0 * 16)
            
        img_flattened = img_array.flatten()
        
        # Make prediction
        prediction = model.predict([img_flattened])
        prediction_proba = model.predict_proba([img_flattened])
        confidence = np.max(prediction_proba) * 100
        
        # Display Results
        st.success(f"### Prediction: Number {prediction[0]}")
        st.info(f"Confidence: {confidence:.2f}%")
        
        # Show probability breakdown across all digits
        st.write("### Confidence Breakdown:")
        st.bar_chart({str(i): prob for i, prob in enumerate(prediction_proba[0])})
            
    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
