import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from openai import OpenAI

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Chest X-Ray AI System",
    page_icon="🫁",
    layout="wide"
)

# ---------------- THEME ----------------
st.markdown("""
<style>
.main { background-color: #0b1220; color: white; }
h1, h2, h3 { color: #38bdf8; }

.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}

[data-testid="stSidebar"] {
    background-color: #0f172a;
}
</style>
""", unsafe_allow_html=True)

# ---------------- OPENAI CLIENT ----------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------- MODEL ----------------
MODEL_URL = "https://huggingface.co/datasets/yamram/xray-model/resolve/main/best_model_final_fixed%20(1).keras"

@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL)
    model_file = BytesIO(response.content)
    return tf.keras.models.load_model(model_file)

model = load_model()

# ---------------- LABELS ----------------
CLASS_NAMES = ["Normal", "Pneumonia"]

# ---------------- PREPROCESS ----------------
def preprocess(image):
    img = image.resize((224, 224))
    img = np.array(img)

    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# ---------------- OPENAI EXPLANATION ----------------
def get_ai_explanation(label, confidence):
    prompt = f"""
You are a medical AI assistant.

A chest X-ray model predicted:
- Diagnosis: {label}
- Confidence: {confidence:.2f}%

Give:
1. Simple explanation
2. Medical interpretation (non-diagnostic)
3. Possible next steps for patient
4. No prescriptions, keep safe language
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a careful medical explanation assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# ---------------- SIDEBAR ----------------
st.sidebar.title("🫁 Chest X-Ray AI")
page = st.sidebar.radio("Navigation", ["🏠 Home", "🤖 AI Doctor", "ℹ️ Info"])

st.sidebar.caption("⚠️ Educational tool only")

# ---------------- INFO PAGE ----------------
if page == "ℹ️ Info":
    st.title("ℹ️ System Info")
    st.write("Deep learning + OpenAI-powered medical explanation system.")
    st.code(MODEL_URL)

# ---------------- AI DOCTOR PAGE ----------------
elif page == "🤖 AI Doctor":
    st.title("🤖 AI Medical Assistant")
    st.write("Ask general questions about chest X-rays")

    q = st.text_input("Ask a question")

    if q:
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful medical education assistant."},
                    {"role": "user", "content": q}
                ]
            )

        st.success(response.choices[0].message.content)

# ---------------- MAIN APP ----------------
else:

    st.title("🫁 Chest X-Ray AI Diagnostic System")
    st.caption("AI-powered medical imaging analysis system")

    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        if st.button("🔍 Analyze X-Ray", use_container_width=True):

            with st.spinner("Analyzing with AI model..."):
                processed = preprocess(image)
                prediction = model.predict(processed)

                predicted_class = np.argmax(prediction)
                confidence = float(np.max(prediction) * 100)
                label = CLASS_NAMES[predicted_class]

            st.markdown("---")

            # ---------------- RESULT ----------------
            st.subheader("🧾 Diagnosis Result")

            if label == "Normal":
                st.success(f"Result: {label}")
            else:
                st.error(f"Result: {label}")

            st.metric("Confidence", f"{confidence:.2f}%")

            # ---------------- CHART ----------------
            st.subheader("📊 Confidence Breakdown")

            fig, ax = plt.subplots()
            ax.bar(CLASS_NAMES, prediction[0])
            ax.set_ylim([0, 1])
            st.pyplot(fig)

            # ---------------- OPENAI EXPLANATION ----------------
            st.subheader("🤖 AI Explanation (OpenAI)")

            with st.spinner("Generating medical explanation..."):
                explanation = get_ai_explanation(label, confidence)

            st.info(explanation)

            # ---------------- REPORT ----------------
            st.subheader("📄 Download Report")

            report = f"""
Chest X-Ray AI Report

Diagnosis: {label}
Confidence: {confidence:.2f}%

Disclaimer: Educational use only, not medical diagnosis.
"""

            st.download_button(
                "Download Report",
                report,
                file_name="xray_report.txt"
            )
