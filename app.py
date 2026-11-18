import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Classifier",
    page_icon="🔢",
    layout="centered"
)

# Load a robust, pre-trained handwritten digit model from Hugging Face with error handling
@st.cache_resource
def load_model_and_processor():
    model_id = "farleyknight/mnist-digit-classification-vit"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    return processor, model

st.title("🔢 Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9).")

# Load model safely
with st.spinner("Loading pre-trained AI model... Please wait."):
    try:
        processor, model = load_model_and_processor()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.error(f"Failed to load model from Hugging Face: {e}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    try:
        # Preprocess image for the vision model
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        inputs = processor(images=image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            
        prediction = torch.argmax(probs).item()
        confidence = probs[prediction].item() * 100
        
        # Display Results
        st.success(f"### Prediction: Number {prediction}")
        st.info(f"Confidence: {confidence:.2f}%")
        
        # Show probability breakdown across all digits
        st.write("### Confidence Breakdown:")
        prob_dict = {str(i): probs[i].item() for i in range(10)}
        st.bar_chart(prob_dict)
            
    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
elif uploaded_file is not None and not model_loaded:
    st.error("Cannot process image because the model failed to load.")
