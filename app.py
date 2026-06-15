import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
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

# 2. Download and Cache Model Weights from Hugging Face
@st.cache_resource
def load_trained_xray_model():
    model_path = 'best_model_final.keras'
    
    # Download the model if it doesn't exist on the server
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        hf_url = "https://huggingface.co/datasets/yamram/xray-model/resolve/main/best_model_final.keras"
        
        with st.spinner("Downloading model weights from Hugging Face Cloud storage... (This takes a moment on first load)"):
            response = requests.get(hf_url, stream=True)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error(f"Failed to download model from Hugging Face. Status Code: {response.status_code}")
                st.stop()
                
    try:
        # Standard load attempt
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    except Exception:
        # BULLETPROOF FALLBACK: Treat the model file directly as a deployment layer 
        # to bypass all Keras internal version object deserialization errors.
        return tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")

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
        predictions_output = model(input_tensor)
        
        # Extract predictions correctly whether loaded as a full model or an inference layer
        if isinstance(predictions_output, dict):
            # TFSMLayer outputs predictions wrapped inside a dictionary matching its output node name
            key = list(predictions_output.keys())[0]
            raw_predictions = predictions_output[key].numpy()
        else:
            raw_predictions = predictions_output if hasattr(predictions_output, 'numpy') else np.array(predictions_output)
            if hasattr(raw_predictions, 'numpy'):
                raw_predictions = raw_predictions.numpy()
                
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
