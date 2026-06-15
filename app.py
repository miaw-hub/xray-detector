import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

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

# 2. Optimized Model Cache Loader (Keeps it running fast on the server)
@st.cache_resource
def load_trained_xray_model():
    # Looks for your file in the root directory of your project
    return tf.keras.models.load_model('best_model.keras')

with st.spinner("Warming up Keras inference layer..."):
    model = load_trained_xray_model()

# 3. File Input Interaction Node
uploaded_file = st.file_uploader("Upload Chest X-Ray (Accepts: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image content
    image = Image.open(uploaded_file)
    st.image(image, caption="Loaded Input Specimen", use_container_width=True)
    
    # 4. Normalized Preprocessing Sequence (Matches your 128x128 training configuration)
    with st.spinner("Formatting image channels and executing inference matrix..."):
        target_shape = (128, 128)
        
        # Ensure image has 3 color channels (RGB)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        img_resized = image.resize(target_shape)
        img_matrix = np.array(img_resized)
        
        # Rescaling normalization factor 
        img_scaled = img_matrix / 255.0  
        
        # Insert structural batch axis: (1, 128, 128, 3)
        input_tensor = np.expand_dims(img_scaled, axis=0)
        
        # 5. Core Model Processing block
        raw_predictions = model.predict(input_tensor)
        
        # Class labels sequence matching your alphabetical directory structure
        target_labels = ['COVID-19', 'Normal', 'Pneumonia']
        
        winning_index = np.argmax(raw_predictions[0])
        final_prediction = target_labels[winning_index]
        confidence_metric = raw_predictions[0][winning_index] * 100

        # 6. User Presentation Interface
        st.write("---")
        st.subheader("Classification Outcome")
        
        # Change colors based on the diagnostic result
        if final_prediction == 'Normal':
            st.success(f"Classification Vector: **{final_prediction}**")
        elif final_prediction == 'COVID-19':
            st.error(f"Classification Vector: **{final_prediction}**")
        else:
            st.warning(f"Classification Vector: **{final_prediction}**")
            
        st.metric(label="Inference Confidence", value=f"{confidence_metric:.2f}%")
        
        # Expandable dropdown to view percentages for all 3 conditions
        with st.expander("Show internal class distribution metrics"):
            for class_name, individual_probability in zip(target_labels, raw_predictions[0]):
                st.write(f"**{class_name}**: {individual_probability*100:.2f}%")