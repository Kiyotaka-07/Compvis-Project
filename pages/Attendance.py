import streamlit as st
import cv2
import pickle
import numpy as np
from datetime import datetime
import os
import threading
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

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

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2rem, 5vw, 2.9rem);
    color: #1e1b4b;
    line-height: 1.18;
    margin-bottom: 0.6rem;
    font-weight: 400;
}
.hero-title span { font-style: italic; color: #6366f1; }
.hero-sub { color: #64748b; font-size: 1rem; line-height: 1.65; max-width: 520px; margin-bottom: 1.8rem; }

.status-pill {
    display: inline-flex; align-items: center; gap: 7px; padding: 6px 14px;
    border-radius: 999px; font-size: 0.8rem; font-weight: 600; margin-bottom: 1.4rem;
}
.status-pill.active { background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.25); }
.status-pill.inactive { background: rgba(148,163,184,0.12); color: #94a3b8; border: 1px solid rgba(148,163,184,0.2); }
.status-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.status-dot.active {
    background: #10b981; box-shadow: 0 0 0 3px rgba(16,185,129,0.2);
    animation: pulse-dot 1.8s ease-in-out infinite;
}
.status-dot.inactive { background: #cbd5e1; }
@keyframes pulse-dot { 0%, 100% { box-shadow: 0 0 0 3px rgba(16,185,129,0.2); } 50% { box-shadow: 0 0 0 6px rgba(16,185,129,0.08); } }

.camera-placeholder {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 14px; border: 2px dashed #e2e8f0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; min-height: 260px; gap: 0.75rem;
}
.camera-placeholder-icon { font-size: 3rem; opacity: 0.35; }
.camera-placeholder-text { font-size: 0.88rem; color: #94a3b8; font-weight: 500; }

div.stButton > button {
    border-radius: 12px !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 0.88rem !important; padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important; border: none !important;
}
div.stButton:first-child > button {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important; color: white !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
}
div.stButton:first-child > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important; }
div.stButton:last-child > button { background: #f1f5f9 !important; color: #64748b !important; }
div.stButton:last-child > button:hover { background: #e2e8f0 !important; color: #475569 !important; }

.stats-strip {
    display: flex; gap: 0; background: white; border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.06); overflow: hidden; margin-bottom: 0.7rem; box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.stat-item { flex: 1; padding: 0.9rem 1rem; text-align: center; border-right: 1px solid rgba(0,0,0,0.05); }
.stat-item:last-child { border-right: none; }
.stat-num { font-family: 'DM Serif Display', serif; font-size: 1.1rem; color: #1e1b4b; }
.stat-lbl { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

.back-link {
    display: inline-flex; align-items: center; gap: 6px; color: #6366f1; font-size: 0.84rem;
    font-weight: 600; text-decoration: none; padding: 6px 0; opacity: 0.85; transition: opacity 0.15s;
}
.back-link:hover { opacity: 1; }
</style>
""", unsafe_allow_html=True)

def load_models():
    # Ensure absolute pathing for reliable execution
    base_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_path)
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(base_dir, "face_model.yml"))
    
    with open(os.path.join(base_dir, "labels.pkl"), "rb") as f:
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
    st.rerun()
if stop:
    st.session_state.run_camera = False
    st.rerun()

st.markdown("""
<div class="stats-strip">
  <div class="stat-item">
    <div class="stat-num">Live Logging Enabled</div>
    <div class="stat-lbl">Check attendance_log.txt</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Thread-safe logging mechanism
if 'logged_users' not in st.session_state:
    st.session_state.logged_users = set()

log_lock = threading.Lock()
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if w < 120 or h < 120:
            continue

        face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        label, confidence = recognizer.predict(face_roi)

        if confidence < 70:
            name = label_dict.get(label, "Unknown")
            text = f"{name} ({int(confidence)})"
            color = (0, 255, 0) # BGR
            
            with log_lock:
                if name not in st.session_state.logged_users:
                    # Write absolutely to root directory
                    base_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
                    with open(os.path.join(base_dir, "attendance_log.txt"), "a") as log_f:
                        log_f.write(f"{name}, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    st.session_state.logged_users.add(name)
        else:
            text = "Unknown"
            color = (0, 0, 255) # BGR

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x, y - th - 14), (x + tw + 10, y), color, -1)
        cv2.putText(img, text, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

if not st.session_state.run_camera:
    st.markdown("""
    <div class="camera-placeholder">
        <div class="camera-placeholder-icon">📷</div>
        <div class="camera-placeholder-text">Camera preview is stopped</div>
    </div>
    """, unsafe_allow_html=True)
else:
    webrtc_streamer(
        key="attendance_streamer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
