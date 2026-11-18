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
st.write("Upload an image of a handwritten digit (0-9).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    try:
        # 1. Convert image to RGB if it has transparency (RGBA), then to Grayscale
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        
        img_gray = image.convert('L')
        
        # 2. Invert colors if it's a light background with dark/colored ink
        # Sklearn digits expect a dark background with light ink (values 0 to 16)
        img_inverted = ImageOps.invert(img_gray)
        
        # 3. Resize to 8x8 using high-quality LANCZOS resampling
        img_resized = img_inverted.resize((8, 8), Image.Resampling.LANCZOS)
        
        # 4. Convert to numpy array and scale properly to sklearn's 0-16 scale
        img_array = np.array(img_resized, dtype=float)
        
        # Normalize to 0-16 range used by load_digits()
        img_normalized = (img_array / 255.0) * 16.0
        img_flattened = img_normalized.flatten()
        
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
