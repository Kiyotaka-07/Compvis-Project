import streamlit as st
import cv2
import os
import pickle
import time
import sys

# Ensure root is in path so we can cleanly import train_model
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train_mode import train_model

DATASET_DIR = os.path.join(ROOT, "dataset")
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_path)

st.set_page_config(
    page_title="Register Student — FaceAttend",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: linear-gradient(160deg, #f0f4ff 0%, #faf5ff 50%, #f0fdf4 100%) !important; min-height: 100vh; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 2rem; max-width: 780px; }

.hero-title { font-family: 'DM Serif Display', serif; font-size: clamp(2rem, 5vw, 2.9rem); color: #1e1b4b; line-height: 1.18; margin: 0.4rem 0 0.6rem 0; font-weight: 400; }
.hero-title span { font-style: italic; color: #10b981; }
.hero-sub { color: #475569; font-size: 1rem; font-weight: 400; line-height: 1.65; max-width: 520px; margin-bottom: 1.8rem; }

.stApp > div, section[data-testid="stSidebar"], .main > div { background: transparent !important; }
div[data-testid="stTextInput"] label { color: #111827 !important; font-size: 0.95rem !important; font-weight: 700 !important; opacity: 1 !important; }
div[data-testid="stTextInput"] label p { color: #111827 !important; font-size: 0.95rem !important; font-weight: 700 !important; margin: 0 !important; }
div.row-widget.stTextInput > label { color: #111827 !important; }

div[data-testid="stTextInput"] input { border-radius: 12px !important; border: 2px solid #cbd5e1 !important; font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important; padding: 0.75rem 1rem !important; background: #ffffff !important; color: #1e1b4b !important; -webkit-text-fill-color: #1e1b4b !important; }
div[data-testid="stTextInput"] input:focus { border-color: #10b981 !important; box-shadow: 0 0 0 4px rgba(16,185,129,0.15) !important; background: #f0fdf4 !important; color: #1e1b4b !important; -webkit-text-fill-color: #1e1b4b !important; }
div[data-testid="stTextInput"] > div { background: transparent !important; }
div[data-testid="stTextInput"] > div > div { background: transparent !important; border: none !important; box-shadow: none !important; }

.camera-placeholder { background: #f8fafc; border-radius: 14px; border: 2px dashed #cbd5e1; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 240px; gap: 0.75rem; }
.camera-placeholder-icon { font-size: 3rem; opacity: 0.35; }
.camera-placeholder-text { font-size: 0.88rem; color: #94a3b8; font-weight: 500; }

div.stButton > button { border-radius: 12px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; font-size: 0.88rem !important; padding: 0.65rem 1.4rem !important; transition: all 0.2s ease !important; width: 100% !important; cursor: pointer !important; }
div.stButton:nth-child(1) > button { background: linear-gradient(135deg, #10b981, #34d399) !important; color: white !important; border: none !important; box-shadow: 0 4px 14px rgba(16,185,129,0.28) !important; }
div.stButton:nth-child(1) > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(16,185,129,0.38) !important; }
div.stButton:nth-child(2) > button { background: #f1f5f9 !important; color: #475569 !important; border: 1px solid #e2e8f0 !important; }
div.stButton:nth-child(2) > button:hover { background: #e2e8f0 !important; color: #334155 !important; }

.step-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1.6rem; flex-wrap: nowrap; }
.step-chip { display: inline-flex; align-items: center; gap: 5px; padding: 5px 14px; border-radius: 999px; font-size: 0.76rem; font-weight: 600; white-space: nowrap; }
.step-chip.done { background: rgba(16,185,129,0.12); color: #059669; border: 1px solid rgba(16,185,129,0.2); }
.step-chip.active { background: #d1fae5; color: #059669; border: 2px solid #10b981; font-weight: 700; }
.step-chip.idle { background: #f1f5f9; color: #94a3b8; border: 1px solid #e2e8f0; }
.step-divider { flex: 1; height: 1px; background: #e2e8f0; min-width: 20px; }

.back-link { display: inline-flex; align-items: center; gap: 6px; color: #10b981; font-size: 0.84rem; font-weight: 600; text-decoration: none; padding: 6px 0; margin-bottom: 0.5rem; opacity: 0.85; transition: opacity 0.15s; }
.back-link:hover { opacity: 1; }
div[data-testid="stImage"] > img { border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

if 'capture_count' not in st.session_state:
    st.session_state.capture_count = 0
if 'registration_complete' not in st.session_state:
    st.session_state.registration_complete = False
if 'start_capture' not in st.session_state:
    st.session_state.start_capture = False
if 'last_capture_time' not in st.session_state:
    st.session_state.last_capture_time = 0

TARGET_IMAGES = 30

st.markdown('<a class="back-link" href="/" target="_self">← Back to Dashboard</a>', unsafe_allow_html=True)
st.markdown("""
<h1 class="hero-title">Register New <span>Student</span></h1>
<p class="hero-sub">
    Fill in the student details below, then let the camera capture
    30 face samples automatically to train the recognition model.
</p>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class="step-row">
  <span class="step-chip {'done' if st.session_state.capture_count >= TARGET_IMAGES else 'active'}">① Fill Details</span>
  <div class="step-divider"></div>
  <span class="step-chip {'done' if st.session_state.registration_complete else ('active' if st.session_state.start_capture else 'idle')}">② Capture Faces</span>
  <div class="step-divider"></div>
  <span class="step-chip {'active' if st.session_state.registration_complete else 'idle'}">③ Registered</span>
</div>
""", unsafe_allow_html=True)

name = st.text_input("Full Name", placeholder="e.g. Budi Santoso", key="name_input")
nim  = st.text_input("NIM (Student ID)", placeholder="e.g. 2024001234", key="nim_input")

if name and nim:
    user_folder = f"{name}_{nim}"
    full_path   = os.path.join(DATASET_DIR, user_folder)
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Auto-Capture"):
            st.session_state.start_capture = True
            st.session_state.capture_count = 0
    with col2:
        if st.button("Reset"):
            st.session_state.capture_count = 0
            st.session_state.registration_complete = False
            st.session_state.start_capture = False

    FRAME_WINDOW = st.empty()

    if not st.session_state.start_capture:
        st.markdown("""
        <div class="camera-placeholder">
            <div class="camera-placeholder-icon">🎥</div>
            <div class="camera-placeholder-text">Camera preview will appear here</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.start_capture:
        cap = cv2.VideoCapture(0)
        MIN_FACE_SIZE = 120
        start_time = time.time()

        try:
            while st.session_state.start_capture and st.session_state.capture_count < TARGET_IMAGES:
                # 60-second timeout to prevent indefinite hanging
                if time.time() - start_time > 60:
                    st.error("Capture timed out. Please face the camera and try again.")
                    st.session_state.start_capture = False
                    break
                
                # Sleep briefly so Streamlit UI thread doesn't lock up
                time.sleep(0.05)
                
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam. Is another app using it?")
                    st.session_state.start_capture = False
                    break

                mirror_frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(mirror_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                        continue

                    current_time = time.time()
                    if current_time - st.session_state.last_capture_time < 0.2:  
                        continue
                    
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (200, 200))

                    st.session_state.capture_count += 1
                    st.session_state.last_capture_time = current_time 
                    img_name = os.path.join(full_path, f"{st.session_state.capture_count}.jpg")
                    cv2.imwrite(img_name, face)

                    color_bgr = (16, 185, 89)
                    cv2.rectangle(mirror_frame, (x, y), (x+w, y+h), color_bgr, 2)
                    label = f"{st.session_state.capture_count}/{TARGET_IMAGES}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(mirror_frame, (x, y - th - 14), (x + tw + 10, y), color_bgr, -1)
                    cv2.putText(mirror_frame, label, (x + 5, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                frame_rgb = cv2.cvtColor(mirror_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb, use_column_width=True)

                if st.session_state.capture_count >= TARGET_IMAGES:
                    st.session_state.start_capture = False
                    st.session_state.registration_complete = True
        finally:
            # Guarantee the camera is released to prevent hardware locks
            cap.release()

    if st.session_state.registration_complete:
        if st.button("Complete Registration"):
            with st.spinner("Processing images — this may take a moment…"):
                try:
                    train_model()
                    st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    border-radius: 18px; padding: 2rem; text-align: center;
    box-shadow: 0 10px 40px rgba(16, 185, 129, 0.3); margin: 2rem 0; border: 3px solid #059669;
">
    <h2 style="color: white; margin: 0 0 0.5rem 0; font-size: 1.8rem; font-weight: 700;">
        Registration Successful!
    </h2>
    <p style="color: #dcfce7; margin: 0.5rem 0 0; font-size: 1.1rem; font-weight: 500;">
        <strong>{name}</strong> ({nim}) has been added to the recognition system.
    </p>
</div>
""", unsafe_allow_html=True)
                except ValueError as e:
                    st.error(f"Cannot complete registration: {e}")
                    
            st.session_state.capture_count = 0
            st.session_state.registration_complete = False
            st.session_state.start_capture = False
