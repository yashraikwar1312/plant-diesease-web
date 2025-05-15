import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import requests
import os
import time

# Set page config
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

# Load custom styles
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Disease info
disease_info = {
    "Apple___Apple_scab": {
        "name": "Apple Scab",
        "description": "Dark lesions on leaves and fruit.",
        "cure": "Apply fungicides and remove infected leaves.",
        "image_url": "https://www.tree-care.info/wp-content/uploads/2020/08/apple-scab.jpg"
    },
    "Apple___Black_rot": {
        "name": "Apple Black Rot",
        "description": "Leaf spots and fruit rot.",
        "cure": "Prune infected branches. Use fungicide.",
        "image_url": "https://extension.umn.edu/sites/extension.umn.edu/files/black-rot-apple.jpg"
    },
    "Corn___Common_rust": {
        "name": "Corn Rust",
        "description": "Brown pustules on leaves.",
        "cure": "Plant resistant varieties. Use fungicide.",
        "image_url": "https://www.forestryimages.org/images/768x512/5525359.jpg"
    },
    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "description": "Dark spots with rings on leaves.",
        "cure": "Use sulfur fungicide. Rotate crops.",
        "image_url": "https://www.apsnet.org/edcenter/disandpath/fungalasco/pdlessons/Article%20Images/PotatoEarlyBlight.jpg"
    },
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "description": "Dark patches on leaves and fruit.",
        "cure": "Copper fungicide. Remove infected parts.",
        "image_url": "https://www.gardeningknowhow.com/wp-content/uploads/2015/01/late-blight.jpg"
    },
    "Healthy": {
        "name": "Healthy Plant",
        "description": "No disease detected.",
        "cure": "Maintain good care: water, sunlight, and clean soil.",
        "image_url": "https://images.unsplash.com/photo-1636204317617-20fa3182e5b3?q=80&w=1000&auto=format&fit=crop"
    }
}

# Load model from Google Drive
@st.cache_resource
def load_disease_model():
    model_path = "best_model.keras"
    if not os.path.exists(model_path):
        file_id = "1ZFBZ28Ou1LM8GJQewFwyyQOp7v4BrXFr"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
        else:
            st.error("Failed to download model.")
            return None
    return load_model(model_path)

# Preprocess image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict
def predict_disease(model, image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    class_idx = np.argmax(prediction[0])
    class_names = list(disease_info.keys())
    return class_names[class_idx], float(prediction[0][class_idx])

# UI
def main():
    st.markdown("""
    <div class="landing">
        <h1>Hi fellows, I am here to help you!</h1>
        <a href="#detect" class="get-started">Get Started</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div id='detect'></div>", unsafe_allow_html=True)
    st.title("Plant Disease Detection")

    col1, col2 = st.columns(2)
    image = None
    with col1:
        st.subheader("Upload Image")
        uploaded = st.file_uploader("Choose a plant image", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded)
    with col2:
        st.subheader("Use Camera")
        camera_img = st.camera_input("Capture image")
        if camera_img:
            image = Image.open(camera_img)

    if image:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                model = load_disease_model()
                if model:
                    for i in range(100):
                        time.sleep(0.01)
                        st.progress(i + 1)
                    label, confidence = predict_disease(model, image)
                    info = disease_info.get(label, {})
                    st.markdown(f"## Prediction: {info.get('name', 'Unknown')}")
                    st.image(info.get("image_url", ""), width=250)
                    st.markdown(f"**Description:** {info.get('description', 'N/A')}")
                    st.markdown(f"**Cure:** {info.get('cure', 'N/A')}")
                    st.markdown(f"**Confidence:** {confidence*100:.2f}%")

if __name__ == "__main__":
    main()
