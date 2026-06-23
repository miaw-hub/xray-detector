import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import requests
from groq import Groq
from io import BytesIO
import datetime
import pandas as pd
import plotly.express as px

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray AI Diagnosis",
    page_icon="🫁",
    layout="wide"
)

# ─── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0a0f1e; color: #e8eaf6; }

.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a237e 50%, #0d47a1 100%);
    border-radius: 20px; padding: 45px 40px; text-align: center;
    margin-bottom: 28px; border: 1px solid rgba(100,181,246,0.2);
    box-shadow: 0 20px 60px rgba(13,71,161,0.4);
}
.hero h1 { font-size: 2.6rem; font-weight: 700; color: #fff; margin: 0 0 10px 0; }
.hero p  { font-size: 1rem; color: #90caf9; margin: 0; font-weight: 300; }

.card {
    background: linear-gradient(145deg, #0d1b2a, #112240);
    border: 1px solid rgba(100,181,246,0.15); border-radius: 16px;
    padding: 24px; margin-bottom: 18px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.card h3 { color: #64b5f6; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; margin: 0 0 14px 0; }

.result-covid     { background: linear-gradient(135deg,#b71c1c,#e53935); border-radius:12px; padding:18px 24px; text-align:center; color:white; font-size:1.6rem; font-weight:700; letter-spacing:2px; box-shadow:0 8px 24px rgba(229,57,53,0.4); }
.result-normal    { background: linear-gradient(135deg,#1b5e20,#43a047); border-radius:12px; padding:18px 24px; text-align:center; color:white; font-size:1.6rem; font-weight:700; letter-spacing:2px; box-shadow:0 8px 24px rgba(67,160,71,0.4); }
.result-pneumonia { background: linear-gradient(135deg,#e65100,#fb8c00); border-radius:12px; padding:18px 24px; text-align:center; color:white; font-size:1.6rem; font-weight:700; letter-spacing:2px; box-shadow:0 8px 24px rgba(251,140,0,0.4); }

.prob-row    { display:flex; align-items:center; margin-bottom:12px; gap:12px; }
.prob-label  { width:100px; font-size:0.8rem; font-weight:600; color:#90caf9; text-transform:uppercase; letter-spacing:0.5px; }
.prob-bar-bg { flex:1; background:rgba(255,255,255,0.07); border-radius:50px; height:9px; overflow:hidden; }
.prob-bar-fill-covid     { height:100%; background:linear-gradient(90deg,#e53935,#ef9a9a); border-radius:50px; }
.prob-bar-fill-normal    { height:100%; background:linear-gradient(90deg,#43a047,#a5d6a7); border-radius:50px; }
.prob-bar-fill-pneumonia { height:100%; background:linear-gradient(90deg,#fb8c00,#ffcc80); border-radius:50px; }
.prob-pct { width:52px; text-align:right; font-size:0.88rem; font-weight:600; color:#e8eaf6; }

.key-finding {
    background: rgba(100,181,246,0.07); border-left: 3px solid #64b5f6;
    border-radius: 0 8px 8px 0; padding: 10px 16px; margin-bottom: 8px;
    color: #cfd8dc; font-size: 0.9rem;
}
.key-finding strong { color: #64b5f6; }

.explanation-box {
    background: linear-gradient(145deg,#0d1b2a,#112240);
    border-left: 4px solid #64b5f6; border-radius: 0 12px 12px 0;
    padding: 20px 24px; color: #cfd8dc; font-size: 0.95rem;
    line-height: 1.8; margin-top: 10px;
}

.model-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 16px; margin-bottom: 8px;
    background: rgba(255,255,255,0.04); border-radius: 10px;
    border: 1px solid rgba(100,181,246,0.1);
}
.model-name  { font-weight: 600; color: #e8eaf6; font-size: 0.9rem; }
.model-acc   { color: #64b5f6; font-weight: 700; font-size: 0.9rem; }
.model-badge-best { background: #1565c0; color: white; font-size: 0.7rem; padding: 2px 8px; border-radius: 20px; margin-left: 8px; }

.stat-box { background: linear-gradient(145deg,#0d1b2a,#112240); border: 1px solid rgba(100,181,246,0.15); border-radius: 12px; padding: 18px; text-align: center; }
.stat-val  { font-size: 1.8rem; font-weight: 700; color: #64b5f6; }
.stat-lbl  { font-size: 0.75rem; color: #546e7a; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

.disclaimer {
    background: rgba(255,193,7,0.08); border: 1px solid rgba(255,193,7,0.25);
    border-radius: 10px; padding: 12px 18px; color: #ffe082;
    font-size: 0.8rem; text-align: center; margin-top: 24px;
}
.placeholder-box {
    text-align: center; padding: 70px 28px;
    background: linear-gradient(145deg,#0d1b2a,#112240);
    border: 1px dashed rgba(100,181,246,0.2); border-radius: 16px;
}
.stButton > button {
    background: linear-gradient(135deg,#1565c0,#0d47a1) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 12px 28px !important; font-size: 0.95rem !important;
    font-weight: 600 !important; width: 100% !important;
    box-shadow: 0 4px 15px rgba(13,71,161,0.4) !important; margin-top: 12px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#1976d2,#1565c0) !important;
    box-shadow: 0 6px 20px rgba(25,118,210,0.5) !important;
}
.upload-hint { text-align:center; color:#546e7a; font-size:0.82rem; margin-top:6px; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ─────────────────────────────────────────────────
groq_client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)
CLASSES   = ['covid', 'normal', 'pneumonia']
ICONS     = {'covid': '🦠', 'normal': '✅', 'pneumonia': '⚠️'}
MODEL_URL = "https://huggingface.co/datasets/yamram/xray-model/resolve/main/best_model_final_fixed%20(1).keras"
IMG_SIZE  = 128

MODEL_STATS = {
    'Custom CNN':   {'acc': 63.0,  'covid': 57.0,  'normal': 72.0,  'pneumonia': 60.0},
    'VGG16':        {'acc': 82.89, 'covid': 97.14, 'normal': 91.34, 'pneumonia': 60.17},
    'ResNet50':     {'acc': 83.95, 'covid': 94.51, 'normal': 97.84, 'pneumonia': 59.52},
    'EfficientNet': {'acc': 76.81, 'covid': 79.34, 'normal': 90.48, 'pneumonia': 60.61},
}

if 'history'      not in st.session_state: st.session_state.history = []
if 'total'        not in st.session_state: st.session_state.total = 0
if 'class_counts' not in st.session_state: st.session_state.class_counts = {'covid':0,'normal':0,'pneumonia':0}

# ─── HELPERS ───────────────────────────────────────────────────
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

def preprocess(image):
    import tensorflow as tf
    img = image.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    arr = np.array(img).astype(np.float32)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def get_key_findings(label, preds):
    c, n, p = preds[0]*100, preds[1]*100, preds[2]*100

    findings = {
        "covid": [
            f"Classification most consistent with COVID-19 ({c:.1f}%)",
            f"Model confidence exceeds Normal class probability ({n:.1f}%)",
            "Image contains patterns associated with COVID-related abnormalities",
            "Clinical review is recommended for confirmation"
        ],
        "normal": [
            f"Classification most consistent with a Normal chest X-ray ({n:.1f}%)",
            "No major abnormal classification patterns detected",
            "COVID and Pneumonia probabilities remain comparatively low",
            "Clinical interpretation should still be performed by a professional"
        ],
        "pneumonia": [
            f"Classification most consistent with Pneumonia ({p:.1f}%)",
            f"Model confidence exceeds Normal class probability ({n:.1f}%)",
            "Image contains patterns associated with pneumonia-related abnormalities",
            "Clinical review is recommended for confirmation"
        ]
    }

    return findings[label]

def explain_result(label, confidence, language, patient_name, patient_age):
    patient_info = (
        f"Patient: {patient_name}, Age: {patient_age}"
        if patient_name else ""
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""
{patient_info}

Chest X-ray AI classification result:
Class: {label}
Confidence: {confidence}

Explain in {language}:

1. What this classification means
2. General information about this condition
3. Questions the patient may discuss with a healthcare professional
4. State clearly this is NOT a diagnosis
5. Do NOT prescribe treatments or medications
6. Use compassionate and easy-to-understand language
"""
            }
        ]
    )

    return response.choices[0].message.content

def explain_prediction_reason(label):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""
Explain why an image classifier may predict {label}.

Discuss general visual patterns associated with the class.

Do not claim specific findings in this image.

Keep explanation concise and educational.
"""
            }
        ]
    )

    return response.choices[0].message.content

def risk_level(label, confidence):
    conf = float(confidence.replace("%", ""))

    if label == "normal":
        return "🟢 Low Risk"

    if conf >= 90:
        return "🔴 High Risk"

    if conf >= 70:
        return "🟠 Moderate Risk"

    return "🟡 Review Recommended"

def generate_pdf(patient_name, patient_age, patient_gender, label, conf, preds, explanation, findings):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.units import inch

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=0.8*inch, bottomMargin=0.8*inch)
        styles = getSampleStyleSheet()
        story  = []

        h = lambda t: ParagraphStyle('h', parent=styles['Normal'], fontSize=13, textColor=colors.HexColor('#1565c0'), spaceAfter=6, fontName='Helvetica-Bold')

        story.append(Paragraph("Chest X-Ray AI Diagnosis Report", ParagraphStyle('T', parent=styles['Title'], fontSize=20, textColor=colors.HexColor('#1565c0'), spaceAfter=4)))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1565c0')))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}", ParagraphStyle('d', parent=styles['Normal'], fontSize=9, textColor=colors.grey)))
        story.append(Spacer(1, 14))

        if patient_name:
            story.append(Paragraph("Patient Information", h('')))
            pt = Table([['Name:', patient_name, 'Age:', patient_age], ['Gender:', patient_gender, 'Date:', datetime.datetime.now().strftime('%Y-%m-%d')]], colWidths=[1.2*inch, 2.3*inch, 1*inch, 2*inch])
            pt.setStyle(TableStyle([('FONTSIZE',(0,0),(-1,-1),10),('TEXTCOLOR',(0,0),(0,-1),colors.grey),('TEXTCOLOR',(2,0),(2,-1),colors.grey),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            story.append(pt)
            story.append(Spacer(1, 14))

        rcolors = {'covid':'#e53935','normal':'#43a047','pneumonia':'#fb8c00'}
        story.append(Paragraph("Diagnosis Result", h('')))
        rt = Table([['Predicted Class','Confidence','Model Used'],[label.upper(),conf,'ResNet50']], colWidths=[2*inch,2*inch,2.5*inch])
        rt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1565c0')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('BACKGROUND',(0,1),(0,1),colors.HexColor(rcolors[label])),('TEXTCOLOR',(0,1),(0,1),colors.white),('FONTSIZE',(0,0),(-1,-1),11),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('ALIGN',(0,0),(-1,-1),'CENTER'),('GRID',(0,0),(-1,-1),0.5,colors.lightgrey),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
        story.append(rt)
        story.append(Spacer(1, 14))

        story.append(Paragraph("Class Probabilities", h('')))
        prob_t = Table([['Class','Probability'],['COVID-19',f'{preds[0]*100:.2f}%'],['Normal',f'{preds[1]*100:.2f}%'],['Pneumonia',f'{preds[2]*100:.2f}%']], colWidths=[3*inch,3*inch])
        prob_t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1565c0')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('ALIGN',(0,0),(-1,-1),'CENTER'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f5f5f5'),colors.white]),('GRID',(0,0),(-1,-1),0.5,colors.lightgrey),('FONTSIZE',(0,0),(-1,-1),11),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.append(prob_t)
        story.append(Spacer(1, 14))

        story.append(Paragraph("Key Radiological Findings", h('')))
        for f in findings:
            clean = f.replace('<strong>','').replace('</strong>','')
            story.append(Paragraph(f"• {clean}", ParagraphStyle('f', parent=styles['Normal'], fontSize=10, spaceAfter=5, leftIndent=10)))
        story.append(Spacer(1, 14))

        story.append(Paragraph("AI Medical Explanation", h('')))
        story.append(Paragraph(explanation.replace('\n','<br/>'), ParagraphStyle('e', parent=styles['Normal'], fontSize=10, leading=16, textColor=colors.HexColor('#333333'))))
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        story.append(Spacer(1, 8))
        story.append(Paragraph("DISCLAIMER: This report is generated by an AI system for research and educational purposes only. Not a substitute for professional medical diagnosis.", ParagraphStyle('disc', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=1)))

        doc.build(story)
        buf.seek(0)
        return buf
    except Exception as e:
        return None

# ─── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    st.markdown("#### 👤 Patient Information")
    patient_name   = st.text_input("Patient Name", placeholder="e.g. Sarah Ahmed")
    patient_age    = st.text_input("Age", placeholder="e.g. 35")
    patient_gender = st.selectbox("Gender", ["Select","Female","Male","Other"])
    st.markdown("---")
    st.markdown("#### 🌍 Explanation Language")
    language = st.selectbox("", ["English","Urdu","Arabic","French","Spanish"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("#### 📊 Session Statistics")
    s1, s2 = st.columns(2)
    with s1: st.markdown(f'<div class="stat-box"><div class="stat-val">{st.session_state.total}</div><div class="stat-lbl">Analyzed</div></div>', unsafe_allow_html=True)
    with s2: st.markdown(f'<div class="stat-box"><div class="stat-val">{st.session_state.class_counts["covid"]}</div><div class="stat-lbl">COVID</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    s3, s4 = st.columns(2)
    with s3: st.markdown(f'<div class="stat-box"><div class="stat-val">{st.session_state.class_counts["normal"]}</div><div class="stat-lbl">Normal</div></div>', unsafe_allow_html=True)
    with s4: st.markdown(f'<div class="stat-box"><div class="stat-val">{st.session_state.class_counts["pneumonia"]}</div><div class="stat-lbl">Pneumonia</div></div>', unsafe_allow_html=True)
    if st.session_state.history:
        st.markdown("---")
        st.markdown("#### 🕓 Recent History")
        for h in reversed(st.session_state.history[-5:]):
            st.markdown(f"`{h['time']}` {ICONS[h['label']]} **{h['label'].upper()}** — {h['conf']}")

# ─── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🫁 Chest X-Ray AI Diagnosis</h1>
    <p>Deep learning powered classification · COVID-19 · Pneumonia · Normal</p>
</div>
""", unsafe_allow_html=True)

# ─── TABS ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔬 Diagnosis", "📊 Model Comparison"])

# ════ TAB 1 ════
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="card"><h3>📤 Upload X-Ray Image</h3>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
        st.markdown('<p class="upload-hint">Supported: JPG · JPEG · PNG</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded:
            image    = Image.open(uploaded)
            enhanced = ImageEnhance.Contrast(image).enhance(1.8)
            c1, c2   = st.columns(2)
            with c1: st.image(image,    caption="Original",  use_column_width=True)
            with c2: st.image(enhanced, caption="Enhanced", use_column_width=True)
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

                    st.session_state.total += 1
                    st.session_state.class_counts[label] += 1
                    st.session_state.history.append({'label': label, 'conf': conf, 'time': datetime.datetime.now().strftime('%H:%M')})

                    st.markdown(f'<div class="result-{label}">{ICONS[label]} &nbsp; {label.upper()} &nbsp;·&nbsp; {conf}</div>', unsafe_allow_html=True)
                    st.info(risk_level(label, conf))
                    st.markdown("### 🎯 Model Confidence")
                    st.progress(float(preds[idx]))
                    st.metric("Confidence Score", conf)
                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown('<div class="card"><h3>📊 Class Probabilities</h3>', unsafe_allow_html=True)
                    for i, cls in enumerate(CLASSES):
                        pct = preds[i] * 100
                        st.markdown(f'<div class="prob-row"><div class="prob-label">{cls}</div><div class="prob-bar-bg"><div class="prob-bar-fill-{cls}" style="width:{pct:.1f}%"></div></div><div class="prob-pct">{pct:.1f}%</div></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # ── Pie chart ──────────────────────────────────────────
                    df = pd.DataFrame({"Class": CLASSES, "Probability": preds * 100})
                    fig = px.pie(
                        df,
                        values="Probability",
                        names="Class",
                        hole=0.55,
                        color="Class",
                        color_discrete_map={
                            "covid":     "#e53935",
                            "normal":    "#43a047",
                            "pneumonia": "#fb8c00"
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    findings = get_key_findings(label, preds)
                    st.markdown('<div class="card"><h3>🔍 Key Radiological Findings</h3>', unsafe_allow_html=True)
                    for f in findings:
                        st.markdown(f'<div class="key-finding">{f}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    label_ok = label
                    preds_ok = preds

                except Exception as e:
                    st.error(f"Analysis error: {e}")
                    label_ok = "Unknown"
                    preds_ok = [0, 0, 0]
                    findings = []

            if label_ok != "Unknown":
                with st.spinner("💬 Generating explanation..."):
                    try:
                        explanation = explain_result(label_ok, conf, language, patient_name, patient_age)
                        reasoning   = explain_prediction_reason(label_ok)

                        st.markdown("""
                        <div class="card">
                        <h3>🧠 Why Did The AI Predict This?</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(reasoning)

                        st.markdown(f'<div class="card"><h3>🤖 AI Medical Explanation ({language})</h3><div class="explanation-box">{explanation.replace(chr(10),"<br>")}</div></div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Explanation error: {e}")
                        explanation = "Explanation unavailable."

                st.markdown("<br>", unsafe_allow_html=True)
                with st.spinner("📄 Preparing PDF..."):
                    try:
                        import reportlab
                        pdf = generate_pdf(patient_name, patient_age, patient_gender, label_ok, conf, preds_ok, explanation, findings)
                        if pdf:
                            fname = f"xray_report_{patient_name.replace(' ','_') if patient_name else 'patient'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                            st.download_button("📥 Download Full PDF Report", data=pdf, file_name=fname, mime="application/pdf", use_container_width=True)
                    except ImportError:
                        st.warning("Add `reportlab` to requirements.txt to enable PDF download.")

        elif not uploaded:
            st.markdown('<div class="placeholder-box"><div style="font-size:4rem;margin-bottom:16px;">🫁</div><div style="color:#546e7a;font-size:0.95rem;line-height:1.8;">Upload a chest X-ray on the left<br>then click <strong style="color:#64b5f6">Analyze X-Ray</strong><br>to get AI-powered diagnosis</div></div>', unsafe_allow_html=True)

# ════ TAB 2 ════
with tab2:
    st.markdown('<div class="card"><h3>🏆 Model Performance Comparison</h3>All four models trained on the same dataset — 6939 balanced chest X-ray images.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>📊 Overall Accuracy</h3>', unsafe_allow_html=True)
    for name, stats in MODEL_STATS.items():
        is_best = name == 'ResNet50'
        badge   = '<span class="model-badge-best">★ BEST</span>' if is_best else ''
        st.markdown(f'<div class="model-row"><div><span class="model-name">{name}</span>{badge}</div><div class="prob-bar-bg" style="flex:1;margin:0 16px;"><div class="prob-bar-fill-normal" style="width:{stats["acc"]}%"></div></div><div class="model-acc">{stats["acc"]}%</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>🎯 Per-Class Accuracy Breakdown</h3>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (name, stats) in zip(cols, MODEL_STATS.items()):
        with col:
            border = "border: 1px solid #1565c0;" if name == 'ResNet50' else ""
            st.markdown(f'''
            <div style="background:rgba(13,27,42,0.8);border-radius:12px;padding:16px;text-align:center;{border}">
                <div style="font-weight:700;color:#e8eaf6;margin-bottom:12px;font-size:0.85rem;">{name}{"  ★" if name=="ResNet50" else ""}</div>
                <div style="margin-bottom:8px;"><div style="font-size:0.7rem;color:#90caf9;">COVID</div><div style="font-size:1.2rem;font-weight:700;color:#e53935;">{stats["covid"]}%</div></div>
                <div style="margin-bottom:8px;"><div style="font-size:0.7rem;color:#90caf9;">NORMAL</div><div style="font-size:1.2rem;font-weight:700;color:#43a047;">{stats["normal"]}%</div></div>
                <div><div style="font-size:0.7rem;color:#90caf9;">PNEUMONIA</div><div style="font-size:1.2rem;font-weight:700;color:#fb8c00;">{stats["pneumonia"]}%</div></div>
            </div>''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>✅ Why ResNet50 Was Selected</h3>
        <div class="key-finding"><strong>Highest Overall Accuracy</strong> — 83.95% across all three classes</div>
        <div class="key-finding"><strong>Best COVID Detection</strong> — 94.51% sensitivity, critical for medical screening</div>
        <div class="key-finding"><strong>Best Normal Detection</strong> — 97.84%, minimizing false positives</div>
        <div class="key-finding"><strong>Deep Residual Learning</strong> — Skip connections allow learning of subtle radiological patterns</div>
        <div class="key-finding"><strong>Stable Training</strong> — BatchNorm frozen layers prevented instability during fine-tuning</div>
    </div>""", unsafe_allow_html=True)

# ─── DISCLAIMER ────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
    It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
</div>""", unsafe_allow_html=True)
