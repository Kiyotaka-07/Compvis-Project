import streamlit as st
import cv2
import os
import numpy as np
import sys

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
.stApp > div, section[data-testid="stSidebar"], .main > div { background: transparent !important; }
div[data-testid="stTextInput"] label { color: #111827 !important; font-size: 0.95rem !important; font-weight: 700 !important; opacity: 1 !important; }
div[data-testid="stTextInput"] input { border-radius: 12px !important; border: 2px solid #cbd5e1 !important; font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important; padding: 0.75rem 1rem !important; background: #ffffff !important; color: #1e1b4b !important; }
div.stButton > button { border-radius: 12px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; font-size: 0.88rem !important; padding: 0.65rem 1.4rem !important; transition: all 0.2s ease !important; width: 100% !important; cursor: pointer !important; }
div.stButton > button { background: linear-gradient(135deg, #10b981, #34d399) !important; color: white !important; border: none !important; box-shadow: 0 4px 14px rgba(16,185,129,0.28) !important; }
div.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(16,185,129,0.38) !important; }
a[data-testid="stPageLink-NavLink"] { display: inline-flex; align-items: center; gap: 6px; color: #10b981; font-size: 0.84rem; font-weight: 600; text-decoration: none; padding: 6px 0; opacity: 0.85; transition: opacity 0.15s; margin-bottom: 0.5rem; }
a[data-testid="stPageLink-NavLink"]:hover { opacity: 1; }
a[data-testid="stPageLink-NavLink"] p { font-weight: 600; margin: 0; color: #10b981; }
</style>
""", unsafe_allow_html=True)

st.page_link("main.py", label="← Back to Dashboard")

st.markdown("""
<h1 class="hero-title">Register New <span>Student</span></h1>
<p class="hero-sub">Enter the details below and snap 5 photos. We will automatically generate 30 dataset variations from them!</p>
""", unsafe_allow_html=True)

TARGET_PHOTOS = 5

name = st.text_input("Full Name", placeholder="e.g. Budi Santoso")
nim  = st.text_input("NIM (Student ID)", placeholder="e.g. 2024001234")

if name and nim:
    user_folder = f"{name}_{nim}"
    full_path   = os.path.join(DATASET_DIR, user_folder)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    if 'raw_faces' not in st.session_state:
        st.session_state.raw_faces = []
        
    count = len(st.session_state.raw_faces)
    st.progress(count / TARGET_PHOTOS, text=f"Captured {count} / {TARGET_PHOTOS} required angles")
    
    if count < TARGET_PHOTOS:
        st.info(f"📸 **Snap a photo!** (Angle {count + 1} of 5). Look straight, then slightly left/right for the others.")
        
        # Changing the key forces the camera block to instantly reset for the next photo!
        pic = st.camera_input("Capture Face", key=f"reg_cam_{count}")
        
        if pic is not None:
            bytes_data = pic.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                st.error("No face detected! Please ensure you are well-lit and try again.")
            else:
                # Find largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w, h = faces[0]
                face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                
                st.session_state.raw_faces.append(face_roi)
                st.rerun()
    else:
        st.success("✅ Awesome! We have enough face data.")
        if st.button("Complete Registration & Auto-Train"):
            with st.spinner("Generating 30 images and training AI model..."):
                idx = 1
                for face in st.session_state.raw_faces:
                    # 1. Original
                    cv2.imwrite(os.path.join(full_path, f"{idx}.jpg"), face)
                    idx += 1
                    # 2. Flipped (Mirrored)
                    cv2.imwrite(os.path.join(full_path, f"{idx}.jpg"), cv2.flip(face, 1))
                    idx += 1
                    # 3. Brighter
                    cv2.imwrite(os.path.join(full_path, f"{idx}.jpg"), cv2.convertScaleAbs(face, alpha=1.2, beta=15))
                    idx += 1
                    # 4. Darker
                    cv2.imwrite(os.path.join(full_path, f"{idx}.jpg"), cv2.convertScaleAbs(face, alpha=0.8, beta=-15))
                    idx += 1
                    # 5. Slightly Blurred
                    cv2.imwrite(os.path.join(full_path, f"{idx}.jpg"), cv2.GaussianBlur(face, (5, 5), 0))
                    idx += 1
                    # 6. Zoomed In
                    zoom = cv2.resize(face[10:190, 10:190], (200, 200))
                    cv2.imwrite(os.path.join(full_path, f"{idx}.jpg"), zoom)
                    idx += 1
                
                # Train the dataset
                train_model()
                
            # Clear cache for the next user
            st.session_state.raw_faces = [] 
            
            st.markdown(f"""
<div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 18px; padding: 2rem; text-align: center; box-shadow: 0 10px 40px rgba(16, 185, 129, 0.3); margin: 1rem 0; border: 3px solid #059669;">
<h2 style="color: white; margin: 0 0 0.5rem 0; font-size: 1.8rem; font-weight: 700;">Registration Successful!</h2>
<p style="color: #dcfce7; margin: 0.5rem 0 0; font-size: 1.1rem; font-weight: 500;"><strong>{name}</strong> ({nim}) has been added to the system.</p>
</div>
""", unsafe_allow_html=True)
            if st.button("Register Another Student"):
                st.rerun()
