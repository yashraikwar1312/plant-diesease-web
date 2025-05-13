import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image
import time

# Load model
model = load_model("best_model.keras")

# Disease class names (make sure this matches your model's class order)
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight", "Grape___healthy",
    "Orange___Citrus_greening", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper___Bacterial_spot", "Pepper___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# Suggestions for each disease (simplified examples)
suggestions = {
    "healthy": "This plant is healthy! No action needed. Keep monitoring regularly.",
    "Apple___Apple_scab": "Use fungicides and remove infected leaves.",
    "Potato___Early_blight": "Apply fungicide and practice crop rotation.",
    "Tomato___Late_blight": "Remove infected leaves and apply a copper-based fungicide.",
    # Add more as needed
}

# Custom styling
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Helvetica' !important;
        }
        .healthy-bg {
            background-color: #d9fdd3;
            padding: 2em;
            border-radius: 12px;
        }
        .disease-bg {
            background-color: #ffe6e6;
            padding: 2em;
            border-radius: 12px;
        }
        .fade-in {
            animation: fadeIn 2s ease-in;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🌿 Plant Disease Detector")
st.subheader("Upload or capture a leaf image to identify the disease")

# File upload or camera input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Take a photo")

image = None
if uploaded_file:
    image = Image.open(uploaded_file)
elif camera_image:
    image = Image.open(camera_image)

# If image provided
if image:
    st.image(image, caption="Input Leaf Image", use_column_width=True)
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner('Analyzing image...'):
        time.sleep(1)  # Animate loading
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        confidence = float(np.max(predictions)) * 100
        predicted_class = class_names[predicted_index]

    # Check for health
    is_healthy = "healthy" in predicted_class.lower()
    suggestion = suggestions.get(predicted_class, "Consult an expert for treatment guidance.")
    if is_healthy:
        suggestion = suggestions["healthy"]

    # Show results with color-coded box
    bg_class = "healthy-bg" if is_healthy else "disease-bg"
    st.markdown(f"""
    <div class="{bg_class} fade-in">
        <h2>Prediction: <strong>{predicted_class}</strong></h2>
        <h4>Confidence: {confidence:.2f}%</h4>
        <p><strong>Suggestion:</strong> {suggestion}</p>
    </div>
    """, unsafe_allow_html=True)
