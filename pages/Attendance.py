import streamlit as st
import cv2
import pickle
import numpy as np
from datetime import datetime
import os

st.set_page_config(
    page_title="Take Attendance — FaceAttend",
    page_icon="📋",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #f0f4ff 0%, #faf5ff 50%, #f0fdf4 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 2rem; max-width: 780px; }

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.08);
    color: #6366f1;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2rem, 5vw, 2.9rem);
    color: #1e1b4b;
    line-height: 1.18;
    margin-bottom: 0.6rem;
    font-weight: 400;
}
.hero-title span {
    font-style: italic;
    color: #6366f1;
}
.hero-sub {
    color: #64748b;
    font-size: 1rem;
    font-weight: 400;
    line-height: 1.65;
    max-width: 520px;
    margin-bottom: 1.8rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 1.4rem;
}
.status-pill.active {
    background: rgba(16,185,129,0.1);
    color: #10b981;
    border: 1px solid rgba(16,185,129,0.25);
}
.status-pill.inactive {
    background: rgba(148,163,184,0.12);
    color: #94a3b8;
    border: 1px solid rgba(148,163,184,0.2);
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.status-dot.active {
    background: #10b981;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.2);
    animation: pulse-dot 1.8s ease-in-out infinite;
}
.status-dot.inactive { background: #cbd5e1; }
@keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 0 3px rgba(16,185,129,0.2); }
    50%       { box-shadow: 0 0 0 6px rgba(16,185,129,0.08); }
}

.camera-card {
    background: white;
    border-radius: 22px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 4px 28px rgba(99,102,241,0.08);
    overflow: hidden;
    margin-bottom: 1.4rem;
}
.camera-inner {
    padding: 1.5rem;
}
.camera-placeholder {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 14px;
    border: 2px dashed #e2e8f0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 260px;
    gap: 0.75rem;
}
.camera-placeholder-icon {
    font-size: 3rem;
    opacity: 0.35;
}
.camera-placeholder-text {
    font-size: 0.88rem;
    color: #94a3b8;
    font-weight: 500;
}

div.stButton > button {
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    border: none !important;
}
div.stButton:first-child > button {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
}
div.stButton:first-child > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}
div.stButton:last-child > button {
    background: #f1f5f9 !important;
    color: #64748b !important;
}
div.stButton:last-child > button:hover {
    background: #e2e8f0 !important;
    color: #475569 !important;
}

.stats-strip {
    display: flex;
    gap: 0;
    background: white;
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.06);
    overflow: hidden;
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.stat-item {
    flex: 1;
    padding: 0.9rem 1rem;
    text-align: center;
    border-right: 1px solid rgba(0,0,0,0.05);
}
.stat-item:last-child { border-right: none; }
.stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.45rem;
    color: #1e1b4b;
}
.stat-lbl {
    font-size: 0.7rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
}

div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}

.back-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #6366f1;
    font-size: 0.84rem;
    font-weight: 600;
    text-decoration: none;
    padding: 6px 0;
    opacity: 0.85;
    transition: opacity 0.15s;
}
.back-link:hover { opacity: 1; }

.soft-footer {
    text-align: center;
    color: #cbd5e1;
    font-size: 0.78rem;
    padding-top: 0.5rem;
    margin-top: 0.5rem;
}

div[data-testid="stImage"] > img {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

def load_models():
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_model.yml")
    with open("labels.pkl", "rb") as f:
        label_dict = pickle.load(f)
    return face_cascade, recognizer, label_dict

try:
    face_cascade, recognizer, label_dict = load_models()
except:
    st.error("⚠️ Models not found — make sure face_model.yml and labels.pkl exist.")
    st.stop()

st.markdown('<a class="back-link" href="/" target="_self">← Back to Dashboard</a>', unsafe_allow_html=True)
st.markdown("""
<h1 class="hero-title">Live <span>Attendance</span> Scan</h1>
<p class="hero-sub">
    Point the camera at any student's face. The system detects, recognises,
    and logs attendance automatically — no interaction needed.
</p>
""", unsafe_allow_html=True)

if 'run_camera' not in st.session_state:
    st.session_state.run_camera = True
if 'last_logged' not in st.session_state:
    st.session_state.last_logged = "—"

if st.session_state.run_camera:
    st.markdown('<div class="status-pill active"><span class="status-dot active"></span>Camera active</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-pill inactive"><span class="status-dot inactive"></span>Camera off</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.4, 1.2, 4])
with col1:
    start = st.button("Start Camera")
with col2:
    stop = st.button("Stop")

if start:
    st.session_state.run_camera = True
    st.experimental_rerun()
if stop:
    st.session_state.run_camera = False
    st.experimental_rerun()

st.markdown(f"""
<div class="stats-strip">
  <div class="stat-item">
    <div class="stat-num">{st.session_state.last_logged}</div>
    <div class="stat-lbl">Last Detected</div>
  </div>
</div>
""", unsafe_allow_html=True)

FRAME_WINDOW = st.empty()

if not st.session_state.run_camera:
    st.markdown("""
    <div class="camera-placeholder">
        <div class="camera-placeholder-icon">📷</div>
        <div class="camera-placeholder-text">Camera preview will appear here</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)

if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)
    threshold = 70
    MIN_FACE_SIZE = 120

    while st.session_state.run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        mirror_frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(mirror_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            label, confidence = recognizer.predict(face_roi)

            if confidence < threshold:
                name = label_dict.get(label, "Unknown")
                text = f"{name} ({int(confidence)})"
                color = (16, 185, 129)   

                with open("attendance_log.txt", "a") as log_f:
                    log_f.write(f"{name}, {datetime.now()}\n")

                st.session_state.last_logged = name
            else:
                text = "Unknown"
                color = (239, 68, 68)   

            cv2.rectangle(mirror_frame, (x, y), (x+w, y+h),
                          (color[2], color[1], color[0]), 2)   
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(mirror_frame,
                          (x, y - th - 14), (x + tw + 10, y),
                          (color[2], color[1], color[0]), -1)
            cv2.putText(mirror_frame, text, (x + 5, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        frame_rgb = cv2.cvtColor(mirror_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, use_column_width=True)

    cap.release()
