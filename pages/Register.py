import streamlit as st
import cv2
import os
import time
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train_mode import train_model

DATASET_DIR = os.path.join(ROOT, "dataset")
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

@st.cache_resource
def load_cascade():
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(haar_path)

face_cascade = load_cascade()

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
a[data-testid="stPageLink-NavLink"] { display: inline-flex; align-items: center; gap: 6px; color: #10b981; font-size: 0.84rem; font-weight: 600; text-decoration: none; padding: 6px 0; opacity: 0.85; transition: opacity 0.15s; margin-bottom: 0.5rem; }
a[data-testid="stPageLink-NavLink"]:hover { opacity: 1; }
a[data-testid="stPageLink-NavLink"] p { font-weight: 600; margin: 0; color: #10b981; }
div.stButton > button { border-radius: 12px !important; font-weight: 600 !important; width: 100% !important; }
</style>
""", unsafe_allow_html=True)

st.page_link("main.py", label="← Back to Dashboard")

st.markdown("""
<h1 class="hero-title">Register New <span>Student</span></h1>
<p class="hero-sub">Fill in the student details below to capture their face profile.</p>
""", unsafe_allow_html=True)

name = st.text_input("Full Name", placeholder="e.g. Budi Santoso")
nim  = st.text_input("NIM (Student ID)", placeholder="e.g. 2024001234")

if name and nim:
    user_folder = f"{name}_{nim}"
    full_path   = os.path.join(DATASET_DIR, user_folder)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    cam_mode = st.radio("Select Capture Mode:", ["📱 Mobile Snapshot (Capture 5 distinct photos)", "💻 PC Live Video (Auto-capture 30 frames)"], horizontal=True)

    if "Mobile" in cam_mode:
        st.info("Take 5 clear pictures of your face. Move your head slightly between shots for better AI accuracy.")
        
        if 'mobile_count' not in st.session_state:
            st.session_state.mobile_count = len(os.listdir(full_path)) if os.path.exists(full_path) else 0

        st.progress(min(st.session_state.mobile_count / 5.0, 1.0), text=f"Captured {st.session_state.mobile_count} / 5 photos")

        pic = st.camera_input("Capture Face")
        if pic is not None:
            bytes_data = pic.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                st.warning("No face detected in that photo! Please try again.")
            else:
                for (x, y, w, h) in faces:
                    face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                    st.session_state.mobile_count += 1
                    cv2.imwrite(os.path.join(full_path, f"mobile_{st.session_state.mobile_count}.jpg"), face)
                    st.success(f"Photo {st.session_state.mobile_count} saved successfully!")
                    break # Only save one face per photo

        if st.session_state.mobile_count >= 5:
            if st.button("Complete Registration & Train AI"):
                with st.spinner("Training recognition model..."):
                    train_model()
                    st.success(f"Successfully registered and trained {name}!")

    else:
        TARGET_IMAGES = 30
        st.info("Press Start. The camera will automatically capture 30 frames of your face.")
        
        col1, col2 = st.columns(2)
        if col1.button("▶ Start Auto-Capture"):
            st.session_state.run_live_reg = True
            st.session_state.capture_count = 0
            st.session_state.last_cap_time = time.time()
            
        if col2.button("⏹ Stop & Reset"):
            st.session_state.run_live_reg = False
            st.session_state.capture_count = 0
            if 'reg_cap' in st.session_state:
                st.session_state.reg_cap.release()
                del st.session_state.reg_cap

        if st.session_state.get('run_live_reg', False):
            if 'reg_cap' not in st.session_state:
                st.session_state.reg_cap = cv2.VideoCapture(0)
                
            cap = st.session_state.reg_cap
            frame_window = st.empty()
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    count = st.session_state.capture_count
                    
                    for (x, y, w, h) in faces:
                        if w < 100 or h < 100: continue
                        current_time = time.time()
                        
                        if current_time - st.session_state.last_cap_time > 0.15:
                            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                            count += 1
                            st.session_state.capture_count = count
                            st.session_state.last_cap_time = current_time
                            cv2.imwrite(os.path.join(full_path, f"pc_{count}.jpg"), face)
                        
                        color = (16, 185, 89) if count >= TARGET_IMAGES else (16, 185, 230)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        label = f"{count}/{TARGET_IMAGES}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x, y - th - 14), (x + tw + 10, y), color, -1)
                        cv2.putText(frame, label, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        break 
                        
                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                if st.session_state.capture_count >= TARGET_IMAGES:
                    st.session_state.run_live_reg = False
                    cap.release()
                    del st.session_state.reg_cap
                    st.success("Capture complete! Click below to train the model.")
                else:
                    st.rerun()

        if st.session_state.get('capture_count', 0) >= TARGET_IMAGES:
            if st.button("Complete Registration & Train AI"):
                with st.spinner("Training recognition model..."):
                    train_model()
                    st.success(f"Successfully registered and trained {name}!")
