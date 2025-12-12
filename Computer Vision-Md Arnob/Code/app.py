import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart Farm Security System",
    page_icon="🚜",
    layout="centered"
)

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct path to the model folder
    model_path = os.path.join(current_dir, "model", "animals_cnn_model.h5")

    # Check if file exists
    if not os.path.exists(model_path):
        return None, model_path
        
    model = tf.keras.models.load_model(model_path)
    return model, model_path

# Load the single model
model, debug_path = load_model()

if model is None:
    st.error("⚠️ Error: Model file not found!")
    st.warning(f"Python looked here: {debug_path}")
    st.info("💡 Fix: Create a folder named 'models' next to app.py and put 'animals_cnn_model.h5' inside it.")
    st.stop()

# --- 2. DEFINE CLASSES ---
# 10 Farm Classes
class_names = {
    0: 'Dog (Cane)', 1: 'Horse (Cavallo)', 2: 'Elephant (Elefante)', 
    3: 'Butterfly (Farfalla)', 4: 'Chicken (Gallina)', 5: 'Cat (Gatto)', 
    6: 'Cow (Mucca)', 7: 'Sheep (Pecora)', 8: 'Spider (Ragno)', 
    9: 'Squirrel (Scoiattolo)'
}

# Authorized Farm Animals
safe_animals = ['Dog (Cane)', 'Cat (Gatto)', 'Chicken (Gallina)', 'Cow (Mucca)', 'Sheep (Pecora)', 'Horse (Cavallo)']

# --- 3. PREDICTION FUNCTION ---
def predict_image(img):
    # Resize to 150x150 (Must match training input)
    img_resized = img.resize((150, 150))
    
    # Convert to array and normalize (0-1 range)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    pred_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    label = class_names.get(pred_idx, "Unknown")
    
    return label, confidence

# --- 4. UI LAYOUT ---
st.sidebar.title("🚜 Control Panel")
st.sidebar.success("✅ System Status: Online")
st.sidebar.info("Model: Custom CNN (Animals-10)")

st.title("🚜 Smart Farm Security System")
st.markdown("---")
st.markdown("### 🛡️ Perimeter Intrusion Detection")
st.write("Upload a live feed image to scan for unauthorized animals or threats.")

# File Uploader
uploaded_file = st.file_uploader("Upload Feed Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Live Perimeter Feed', use_column_width=True)
    
    # "Scan" Button
    if st.button("🔍 Scan Area"):
        with st.spinner('Analyzing biological signatures...'):
            label, conf = predict_image(image)
            
            st.markdown("---")
            st.subheader(f"Identified Entity: **{label}**")
            st.caption(f"AI Confidence: {conf:.2f}%")
            
            # LOGIC: Safe vs Intrusion
            if label in safe_animals:
                st.success("✅ STATUS: AUTHORIZED ANIMAL. (Safe)")
            else:
                st.error("🚨 STATUS: INTRUSION DETECTED! (Threat)")

# Footer
st.markdown("---")
st.caption("CMS22202 Assessment | Automated Security Prototype")