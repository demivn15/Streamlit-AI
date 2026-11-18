import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Classifier",
    page_icon="🔢",
    layout="centered"
)

# Define a lightweight, high-accuracy CNN for MNIST
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

@st.cache_resource
def get_trained_model():
    device = torch.device("cpu")
    model = MNISTCNN().to(device)
    
    # Download standard MNIST dataset (60,000 images) and train for 2 quick epochs (~10-15s once)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(2):
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model

def preprocess_user_image(image: Image.Image) -> torch.Tensor:
    # 1. Handle transparency (paste onto white background)
    if image.mode in ('RGBA', 'LA'):
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[-1])
        image = bg

    # 2. Convert to grayscale
    gray = image.convert('L')
    arr = np.array(gray)

    # 3. Invert if background is white (MNIST expects white digit on black background)
    if np.mean(arr) > 127:
        gray = ImageOps.invert(gray)
        arr = np.array(gray)

    # 4. Crop tightly around the drawn digit to remove excess margins
    # Threshold out faint noise/specks
    coords = np.argwhere(arr > 50)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = gray.crop((x0, y0, x1, y1))
        
        # Pad into a square and add ~20% margin (matching MNIST centering format)
        w, h = cropped.size
        max_side = max(w, h)
        pad = int(max_side * 0.25)
        new_size = max_side + 2 * pad
        square = Image.new('L', (new_size, new_size), 0)
        square.paste(cropped, ((new_size - w) // 2, (new_size - h) // 2))
        processed_img = square.resize((28, 28), Image.Resampling.LANCZOS)
    else:
        processed_img = gray.resize((28, 28), Image.Resampling.LANCZOS)

    # 5. Apply MNIST normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(processed_img).unsqueeze(0), processed_img

st.title("🔢 Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit (0–9).")

with st.spinner("Initializing MNIST CNN model (takes ~15 seconds on first boot)..."):
    model = get_trained_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    tensor_input, preview_img = preprocess_user_image(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(preview_img, caption="Model Input (28×28 Centered)", use_container_width=True)

    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    prediction = torch.argmax(probs).item()
    confidence = probs[prediction].item() * 100

    st.success(f"### Prediction: Number {prediction}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.write("### Confidence Breakdown:")
    prob_dict = {str(i): probs[i].item() for i in range(10)}
    st.bar_chart(prob_dict)
