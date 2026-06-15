import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.request
import os

# 1. Page Configuration & Title Dashboard
st.set_page_config(page_title="X-Ray Classification Hub", layout="centered")
st.title("🩻 Chest X-Ray Diagnostic Classifier")
st.write("Upload a patient's chest X-ray image to identify features pointing to Normal, COVID-19, or Pneumonia conditions.")

# Performance metrics matching your notebook attributes
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Model Validation Accuracy", value="84.00%")
with col2:
    st.metric(label="Target Input Resolution", value="128x128 px")

# 2. Download and Cache Model from Hugging Face
@st.cache_resource
def load_trained_xray_model():
    # The name the model will be saved as locally on the Streamlit server
    model_path = 'best_model_final.keras'
    
    # If the model isn't downloaded onto the server yet, grab it from Hugging Face
    if not os.path.exists(model_path):
        hf_url = "https://huggingface.co/datasets/yamram/xray-model/resolve/main/best_model_final.keras"
        with st.spinner("Downloading model weights from Hugging Face Cloud storage... (This takes a moment on first load)"):
            urllib.request.urlretrieve(hf_url, model_path)
            
    # FIX: Bypass strict layer structural checking rules in Keras v3 envs
    return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

with st.spinner("Warming up Keras inference layer..."):
    model = load_trained_xray_model()

# 3. File Input Interaction Node
uploaded_file = st.file_uploader("Upload Chest X-Ray (Accepts: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Loaded Input Specimen", use_container_width=True)
    
    # 4. Normalized Preprocessing Sequence (128x128)
    with st.spinner("Formatting image channels and executing inference matrix..."):
        target_shape = (128, 128)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        img_resized = image.resize(target_shape)
        img_matrix = np.array(img_resized)
        img_scaled = img_matrix / 255.0  
        input_tensor = np.expand_dims(img_scaled, axis=0)
        
        # 5. Core Model Processing block
        raw_predictions = model.predict(input_tensor)
        target_labels = ['COVID-19', 'Normal', 'Pneumonia']
        
        winning_index = np.argmax(raw_predictions[0])
        final_prediction = target_labels[winning_index]
        confidence_metric = raw_predictions[0][winning_index] * 100

        # 6. User Presentation Interface
        st.write("---")
        st.subheader("Classification Outcome")
        
        if final_prediction == 'Normal':
            st.success(f"Classification Vector: **{final_prediction}**")
        elif final_prediction == 'COVID-19':
            st.error(f"Classification Vector: **{final_prediction}**")
        else:
            st.warning(f"Classification Vector: **{final_prediction}**")
            
        st.metric(label="Inference Confidence", value=f"{confidence_metric:.2f}%")
        
        with st.expander("Show internal class distribution metrics"):
            for class_name, individual_probability in zip(target_labels, raw_predictions[0]):
                st.write(f"**{class_name}**: {individual_probability*100:.2f}%")
