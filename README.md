# 🔢 Handwritten Digit Classification App

A web application built with **Streamlit**, **PyTorch**, and **Hugging Face Transformers** that classifies handwritten digits (`0`–`9`) from uploaded images. It uses a pre-trained Vision Transformer (`ViT`) model fine-tuned on the MNIST dataset and provides real-time confidence breakdowns across all ten digit classes.

## 🚀 Live Demo & Deployment

This project is designed for seamless deployment on **Streamlit Community Cloud** directly from GitHub:

1. Push this repository to your GitHub account.
2. Connect your GitHub repository to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Set the main file path to `app.py` and click **Deploy**.

## 🧠 Model & Architecture Overview

* **Model Architecture:** Pre-trained Vision Transformer (`farleyknight/mnist-digit-classification-vit`) loaded via Hugging Face `transformers`.
* **Dataset:** Fine-tuned on the **MNIST** handwritten digit dataset (`0`–`9`), allowing robust recognition of real-world user drawings without requiring local training or bundled `.joblib` weights.
* **Image Preprocessing Pipeline:**
  * Converts uploaded images (`.jpg`, `.jpeg`, `.png`) to standard 3-channel `RGB` format.
  * Automatically resizes, normalizes, and formats tensors using `AutoImageProcessor` to match the Vision Transformer's expected input dimensions.
  * Runs lightweight CPU inference via PyTorch (`torch.no_grad()`) and applies softmax to compute class probabilities.

## 📁 Project Structure

```text
├── app.py                # Main Streamlit frontend & Hugging Face inference pipeline
├── requirements.txt      # Python dependencies (configured for CPU-only PyTorch)
└── README.md             # Project documentation
```

## 🛠️ Local Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install Dependencies (CPU-Only)

Ensure you have Python 3.9+ installed. The `requirements.txt` file is configured to pull lightweight CPU-only builds of PyTorch to keep installation fast and memory-efficient:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`. On the first run, the pre-trained model weights will be downloaded and cached automatically.

## 📸 Usage Tips for Best Accuracy

* Upload clear images (`.jpg`, `.jpeg`, or `.png`) containing a single handwritten digit (`0`–`9`).
* Review the **Confidence Breakdown** bar chart below each prediction to inspect how the model weighs probabilities across all ten digit classes.


