# 🔢 Handwritten Digit Classification App

A web application built with **Streamlit** and **scikit-learn** that classifies handwritten digits (`0`–`9`) from uploaded images. It features automated image preprocessing, color inversion, transparency handling, and a real-time confidence breakdown chart.

---

## 🚀 Live Demo & Deployment

This project is designed for seamless deployment on **Streamlit Community Cloud** directly from GitHub:
1. Push this repository to your GitHub account.
2. Connect your GitHub repository to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Set the main file path to `app.py` and click **Deploy**.

---

## 🧠 Model & Dataset Overview

* **Model Architecture:** Support Vector Classifier (`SVC`) with RBF kernel (`gamma=0.001`, `C=10.0`, and probability estimation enabled).
* **Dataset:** Trained on scikit-learn's built-in **Digits Dataset** (`sklearn.datasets.load_digits`), which consists of 1,797 grayscale $8 \times 8$ pixel matrices representing handwritten digits from `0` to `9`.
* **Image Preprocessing Pipeline:**
  * Converts transparent (`RGBA`/`LA`) PNGs onto a solid white background.
  * Converts images to grayscale (`L`) and inverts pixel intensity to match the dark-background, light-stroke format of the training data.
  * Downsamples images to $8 \times 8$ resolution using LANCZOS resampling and normalizes pixel intensities to the $0\text{–}16$ scale.

---

## 📁 Project Structure

```text
├── app.py                # Main Streamlit frontend & inference pipeline
├── train.py              # Script to train and export the SVC model
├── model.joblib          # Pre-trained serialized scikit-learn model
├── requirements.txt      # Python dependencies for Streamlit Cloud
└── README.md             # Project documentation
```

---

## 🛠️ Local Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Generate the Model (Optional)
If you need to regenerate `model.joblib` locally:
```bash
python train.py
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
The app will open automatically in your default browser at `http://localhost:8501`.

---

## 📸 Usage Tips for Best Accuracy

* Upload clear images (`.jpg`, `.jpeg`, or `.png`) containing a single centered digit (`0`–`9`).
* Dark or colored ink on a solid white/light background works best, as the app automatically handles color inversion and normalization.
* Review the **Confidence Breakdown** bar chart below each prediction to inspect how the model weighs probabilities across all ten digit classes.
