# 🌿 Plant Disease Detection with Streamlit

A deep learning-powered web app to detect plant leaf diseases using uploaded or camera-captured images. It provides disease predictions, confidence scores, and treatment suggestions — with a clean, animated UI and color-coded results.

---

## 🔍 Features

- Upload or capture leaf images via webcam
- Predict plant disease using a CNN model
- Confidence score for prediction
- Treatment suggestions for each disease
- Green background for healthy leaves
- Red background for diseased leaves
- Clean Helvetica-styled interface with animations

---

## 🧠 Model Info

- Model: Convolutional Neural Network (CNN)
- Format: `best_model.keras`
- Input size: 224x224 RGB images
- Output: 38 classes of plant health and diseases

---

## 📁 Folder Structure
'''project/
│
├── app.py
├── best_model.keras
└── requirements.txt'''
---

## ▶️ How to Run Locally

1. Clone the repository or download the files.

```bash
git clone https://github.com/your-username/plant-disease-detector.git
cd plant-disease-detector

pip install -r requirements.txt
streamlit run app.py'''

📝 # Suggestions Format

Suggestions are shown for known diseases; for others, it prompts to consult an expert. Healthy plants are celebrated with a green background and encouragement.


---

✅ # Example Prediction Output

Prediction: Tomato___Late_blight

Confidence: 94.21%

Suggestion: Remove infected leaves and apply a copper-based fungicide.

Background: Light red for disease



📸 # Try It With Your Plants!

Upload a clear, close-up image of a leaf

Or use the camera directly in the app

Get instant insights for smart agriculture


©️ License

This project is open for educational and research purposes. Please cite or credit the author when reused.
