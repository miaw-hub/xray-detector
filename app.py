import streamlit as st
import keras
from PIL import Image
import numpy as np
import requests
import os

st.set_page_config(page_title="X-Ray Classification Hub", layout="centered")
st.title("🩻 Chest X-Ray Diagnostic Classifier")
st.write("Upload a patient's chest X-ray image to identify Normal, COVID-19, or Pneumonia.")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Model Validation Accuracy", value="84.00%")
with col2:
    st.metric(label="Target Input Resolution", value="128x128 px")

@st.cache_resource
def load_trained_xray_model():
    model_path = '/tmp/best_model_final_fixed.keras'

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        hf_url = "https://huggingface.co/datasets/yamram/xray-model/resolve/main/best_model_final_fixed%20(1).keras"
        with st.spinner("Downloading model from Hugging Face..."):
            response = requests.get(hf_url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error(f"❌ Download failed. Status: {response.status_code}")
                st.stop()

    return keras.saving.load_model(model_path, compile=False)

with st.spinner("Warming up model..."):
    model = load_trained_xray_model()

uploaded_file = st.file_uploader("Upload Chest X-Ray (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    with st.spinner("Running inference..."):
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_resized = image.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        input_tensor = np.expand_dims(img_array, axis=0)

        raw_predictions = model.predict(input_tensor)
        target_labels = ['COVID-19', 'Normal', 'Pneumonia']
        winning_index = np.argmax(raw_predictions[0])
        final_prediction = target_labels[winning_index]
        confidence_metric = raw_predictions[0][winning_index] * 100

        st.write("---")
        st.subheader("Classification Outcome")

        if final_prediction == 'Normal':
            st.success(f"Classification: **{final_prediction}**")
        elif final_prediction == 'COVID-19':
            st.error(f"Classification: **{final_prediction}**")
        else:
            st.warning(f"Classification: **{final_prediction}**")

        st.metric(label="Inference Confidence", value=f"{confidence_metric:.2f}%")

        with st.expander("Show class probability breakdown"):
            for class_name, prob in zip(target_labels, raw_predictions[0]):
                st.write(f"**{class_name}**: {prob * 100:.2f}%")
