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
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(160deg, #f0f4ff 0%, #faf5ff 50%, #f0fdf4 100%); min-height: 100vh; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 2rem; max-width: 780px; }
.hero-title { font-family: 'DM Serif Display', serif; font-size: clamp(2rem, 5vw, 2.9rem); color: #1e1b4b; line-height: 1.18; margin-bottom: 0.6rem; font-weight: 400; }
.hero-title span { font-style: italic; color: #6366f1; }
.hero-sub { color: #64748b; font-size: 1rem; line-height: 1.65; max-width: 520px; margin-bottom: 1.8rem; }
a[data-testid="stPageLink-NavLink"] { display: inline-flex; align-items: center; gap: 6px; color: #6366f1; font-size: 0.84rem; font-weight: 600; text-decoration: none; padding: 6px 0; opacity: 0.85; transition: opacity 0.15s; margin-bottom: 0.5rem; }
a[data-testid="stPageLink-NavLink"]:hover { opacity: 1; }
a[data-testid="stPageLink-NavLink"] p { font-weight: 600; margin: 0; color: #6366f1; }
div.stButton > button { border-radius: 12px !important; font-weight: 600 !important; width: 100% !important; padding: 0.65rem !important; }
</style>
""", unsafe_allow_html=True)

st.page_link("main.py", label="← Back to Dashboard")

@st.cache_resource
def load_models():
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
except Exception as e:
    st.markdown('<h1 class="hero-title">Welcome to <span>FaceAttend!</span></h1>', unsafe_allow_html=True)
    st.info("👋 **It looks like your recognition system hasn't been trained yet.**\n\nThe system needs to know what your students look like before it can take attendance.")
    if st.button("Go to Register Student"): st.switch_page("pages/Register.py")
    st.stop()

st.markdown("""
<h1 class="hero-title">Take <span>Attendance</span></h1>
<p class="hero-sub">Snap a clear photo of the student's face to instantly log their attendance.</p>
""", unsafe_allow_html=True)

if 'logged_today' not in st.session_state:
    st.session_state.logged_today = set()

if 'att_cam_key' not in st.session_state:
    st.session_state.att_cam_key = 0

pic = st.camera_input("Scan Face", key=f"att_cam_{st.session_state.att_cam_key}")

if pic is not None:
    bytes_data = pic.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("No face detected! Please ensure your face is well-lit and try again.")
    else:
        detected = False
        for (x, y, w, h) in faces:
            if w < 80 or h < 80: continue
            
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence = recognizer.predict(face_roi)

            if confidence < 75:
                name = label_dict.get(label, "Unknown")
                text = f"{name} ({int(confidence)})"
                color = (0, 255, 0)
                
                if name not in st.session_state.logged_today:
                    base_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
                    with open(os.path.join(base_dir, "attendance_log.txt"), "a") as log_f:
                        log_f.write(f"{name}, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    st.session_state.logged_today.add(name)
                    st.success(f"✅ Logged attendance for **{name}**!")
                else:
                    st.info(f"ℹ️ **{name}** is already logged for today.")
            else:
                text = "Unknown"
                color = (0, 0, 255)
                st.error("Face not recognized! Please register this student.")

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x, y - th - 14), (x + tw + 10, y), color, -1)
            cv2.putText(img, text, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            detected = True
            break # Only process the most prominent face

        if not detected:
            st.warning("Face too small or unclear. Get closer to the camera.")
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    if st.button("📸 Scan Next Person"):
        st.session_state.att_cam_key += 1
        st.rerun()
