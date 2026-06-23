import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from PIL import Image
import numpy as np
import requests
from openai import OpenAI

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray AI Diagnosis",
    page_icon="🫁",
    layout="wide"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0a0f1e; color: #e8eaf6; }

.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a237e 50%, #0d47a1 100%);
    border-radius: 20px;
    padding: 50px 40px;
    text-align: center;
    margin-bottom: 30px;
    border: 1px solid rgba(100,181,246,0.2);
    box-shadow: 0 20px 60px rgba(13,71,161,0.4);
}
.hero h1 { font-size: 2.8rem; font-weight: 700; color: #ffffff; margin: 0 0 12px 0; letter-spacing: -0.5px; }
.hero p  { font-size: 1.1rem; color: #90caf9; margin: 0; font-weight: 300; }

.card {
    background: linear-gradient(145deg, #0d1b2a, #112240);
    border: 1px solid rgba(100,181,246,0.15);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.card h3 {
    color: #64b5f6;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 0 0 16px 0;
}

.result-covid     { background: linear-gradient(135deg,#b71c1c,#e53935); border-radius:12px; padding:20px 28px; text-align:center; color:white; font-size:1.8rem; font-weight:700; letter-spacing:2px; box-shadow:0 8px 24px rgba(229,57,53,0.4); }
.result-normal    { background: linear-gradient(135deg,#1b5e20,#43a047); border-radius:12px; padding:20px 28px; text-align:center; color:white; font-size:1.8rem; font-weight:700; letter-spacing:2px; box-shadow:0 8px 24px rgba(67,160,71,0.4); }
.result-pneumonia { background: linear-gradient(135deg,#e65100,#fb8c00); border-radius:12px; padding:20px 28px; text-align:center; color:white; font-size:1.8rem; font-weight:700; letter-spacing:2px; box-shadow:0 8px 24px rgba(251,140,0,0.4); }

.prob-row   { display:flex; align-items:center; margin-bottom:14px; gap:12px; }
.prob-label { width:100px; font-size:0.85rem; font-weight:600; color:#90caf9; text-transform:uppercase; letter-spacing:0.5px; }
.prob-bar-bg { flex:1; background:rgba(255,255,255,0.07); border-radius:50px; height:10px; overflow:hidden; }
.prob-bar-fill-covid     { height:100%; background:linear-gradient(90deg,#e53935,#ef9a9a); border-radius:50px; }
.prob-bar-fill-normal    { height:100%; background:linear-gradient(90deg,#43a047,#a5d6a7); border-radius:50px; }
.prob-bar-fill-pneumonia { height:100%; background:linear-gradient(90deg,#fb8c00,#ffcc80); border-radius:50px; }
.prob-pct { width:52px; text-align:right; font-size:0.9rem; font-weight:600; color:#e8eaf6; }

.explanation-box {
    background: linear-gradient(145deg,#0d1b2a,#112240);
    border-left: 4px solid #64b5f6;
    border-radius: 0 12px 12px 0;
    padding: 24px 28px;
    color: #cfd8dc;
    font-size: 0.97rem;
    line-height: 1.8;
    margin-top: 10px;
}

.disclaimer {
    background: rgba(255,193,7,0.08);
    border: 1px solid rgba(255,193,7,0.25);
    border-radius: 10px;
    padding: 14px 20px;
    color: #ffe082;
    font-size: 0.82rem;
    text-align: center;
    margin-top: 30px;
}

.upload-hint { text-align:center; color:#546e7a; font-size:0.85rem; margin-top:8px; }

.stButton > button {
    background: linear-gradient(135deg,#1565c0,#0d47a1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 15px rgba(13,71,161,0.4) !important;
    margin-top: 16px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#1976d2,#1565c0) !important;
    box-shadow: 0 6px 20px rgba(25,118,210,0.5) !important;
}

.placeholder-box {
    text-align: center;
    padding: 80px 28px;
    background: linear-gradient(145deg,#0d1b2a,#112240);
    border: 1px dashed rgba(100,181,246,0.2);
    border-radius: 16px;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ─── KEYS & CONSTANTS ──────────────────────────────────────────
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
CLASSES    = ['covid', 'normal', 'pneumonia']
ICONS      = {'covid': '🦠', 'normal': '✅', 'pneumonia': '⚠️'}
MODEL_URL  = "https://huggingface.co/datasets/yamram/xray-model/resolve/main/best_model_final_fixed%20(1).keras"
IMG_SIZE   = 128   # ← FIX: model was trained on 128x128 not 224x224

# ─── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf
    model_path = "model.keras"
    if not os.path.exists(model_path):
        r = requests.get(MODEL_URL, stream=True)
        with open(model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return tf.keras.models.load_model(model_path, compile=False)

# ─── PREPROCESS ────────────────────────────────────────────────
def preprocess(image):
    import tensorflow as tf
    img = image.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")   # ← 128x128
    arr = np.array(img).astype(np.float32)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ─── OPENAI EXPLANATION ────────────────────────────────────────
def explain_result(label, confidence):
    prompt = f"""
A chest X-ray AI model predicted:
- Result: {label.upper()}
- Confidence: {confidence}

Explain in simple, empathetic medical terms covering:
1. What this result means
2. What the patient might be experiencing
3. Recommended next steps
4. Note: this is AI assistance only, not a final diagnosis

Keep it clear, human, and reassuring. No prescriptions.
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# ─── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🫁 Chest X-Ray AI Diagnosis</h1>
    <p>Deep learning powered classification · COVID-19 · Pneumonia · Normal</p>
</div>
""", unsafe_allow_html=True)

# ─── MAIN LAYOUT ───────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="card"><h3>📤 Upload X-Ray Image</h3>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown('<p class="upload-hint">Supported: JPG · JPEG · PNG</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
        st.button("🔍 Analyze X-Ray", key="analyze_btn")

with right:
    if uploaded and st.session_state.get("analyze_btn"):

        with st.spinner("⏳ Loading AI model..."):
            model = load_model()

        with st.spinner("🔬 Analyzing X-ray..."):
            try:
                arr   = preprocess(image)
                preds = model.predict(arr, verbose=0)[0]
                idx   = int(np.argmax(preds))
                label = CLASSES[idx]
                conf  = f"{preds[idx]*100:.2f}%"

                # Result badge
                st.markdown(f"""
                <div class="result-{label}">
                    {ICONS[label]} &nbsp; {label.upper()} &nbsp; · &nbsp; {conf}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Probabilities
                st.markdown('<div class="card"><h3>📊 Class Probabilities</h3>', unsafe_allow_html=True)
                for i, cls in enumerate(CLASSES):
                    pct = preds[i] * 100
                    st.markdown(f"""
                    <div class="prob-row">
                        <div class="prob-label">{cls}</div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill-{cls}" style="width:{pct:.1f}%"></div>
                        </div>
                        <div class="prob-pct">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Analysis error: {e}")
                label = "Unknown"
                conf  = "N/A"

        # AI Explanation
        if label != "Unknown":
            with st.spinner("💬 Generating medical explanation..."):
                try:
                    explanation = explain_result(label, conf)
                    st.markdown(f"""
                    <div class="card">
                        <h3>🤖 AI Medical Explanation</h3>
                        <div class="explanation-box">{explanation}</div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Explanation error: {e}")

    else:
        st.markdown("""
        <div class="placeholder-box">
            <div style="font-size:5rem; margin-bottom:20px;">🫁</div>
            <div style="color:#546e7a; font-size:1rem; line-height:1.8;">
                Upload a chest X-ray on the left<br>then click <strong style="color:#64b5f6">Analyze X-Ray</strong><br>to get AI-powered results
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── DISCLAIMER ────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
    It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)
