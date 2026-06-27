import streamlit as st
import cv2
import os
import time
import sys
import threading
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from twilio.rest import Client

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train_mode import train_model

DATASET_DIR = os.path.join(ROOT, "dataset")
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

@st.cache_resource
def get_ice_servers():
    """Fetch Twilio TURN servers dynamically to bypass firewalls."""
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        print(f"Twilio error: {e}")
        st.error(f"SYSTEM ERROR: {e}") 
        st.markdown(
            '<div style="background-color: #fff3cd; color: #856404; padding: 1rem; border-radius: 0.5rem; border: 1px solid #ffeeba; margin-bottom: 1rem;">'
            '⚠️ <strong>Warning:</strong> Could not connect to Twilio TURN server. Falling back to free STUN.'
            '</div>', 
            unsafe_allow_html=True
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

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
div[data-testid="stTextInput"] label { color: #111827 !important; font-size: 0.95rem !important; font-weight: 700 !important; opacity: 1 !important; }
div[data-testid="stTextInput"] input { border-radius: 12px !important; border: 2px solid #cbd5e1 !important; font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important; padding: 0.75rem 1rem !important; background: #ffffff !important; color: #1e1b4b !important; -webkit-text-fill-color: #1e1b4b !important; }
div.stButton > button { border-radius: 12px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; font-size: 0.88rem !important; padding: 0.65rem 1.4rem !important; transition: all 0.2s ease !important; width: 100% !important; cursor: pointer !important; }
div.stButton > button { background: linear-gradient(135deg, #10b981, #34d399) !important; color: white !important; border: none !important; box-shadow: 0 4px 14px rgba(16,185,129,0.28) !important; }
div.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(16,185,129,0.38) !important; }
a[data-testid="stPageLink-NavLink"] { display: inline-flex; align-items: center; gap: 6px; color: #10b981; font-size: 0.84rem; font-weight: 600; text-decoration: none; padding: 6px 0; opacity: 0.85; transition: opacity 0.15s; margin-bottom: 0.5rem; }
a[data-testid="stPageLink-NavLink"]:hover { opacity: 1; }
a[data-testid="stPageLink-NavLink"] p { font-weight: 600; margin: 0; color: #10b981; }
</style>
""", unsafe_allow_html=True)

TARGET_IMAGES = 30
st.page_link("main.py", label="← Back to Dashboard")

st.markdown("""
<h1 class="hero-title">Register New <span>Student</span></h1>
<p class="hero-sub">
    Fill in the student details below, then let the camera capture
    30 face samples automatically to train the recognition model.
</p>
""", unsafe_allow_html=True)

name = st.text_input("Full Name", placeholder="e.g. Budi Santoso")
nim  = st.text_input("NIM (Student ID)", placeholder="e.g. 2024001234")

@st.cache_resource
def get_capture_state():
    return {"users": {}, "lock": threading.Lock()}
    
cap_state = get_capture_state()

if name and nim:
    user_folder = f"{name}_{nim}"
    full_path   = os.path.join(DATASET_DIR, user_folder)
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            with cap_state["lock"]:
                if user_folder not in cap_state["users"]:
                    cap_state["users"][user_folder] = {"count": 0, "last_time": 0}
                
                user_data = cap_state["users"][user_folder]

                if user_data["count"] < TARGET_IMAGES:
                    for (x, y, w, h) in faces:
                        if w < 120 or h < 120: continue
                        current_time = time.time()
                        if current_time - user_data["last_time"] > 0.2:
                            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                            user_data["count"] += 1
                            user_data["last_time"] = current_time
                            img_name = os.path.join(full_path, f"{user_data['count']}.jpg")
                            cv2.imwrite(img_name, face)
                            break 
                
                c = user_data["count"]
                color_bgr = (16, 185, 89) if c >= TARGET_IMAGES else (16, 185, 230)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), color_bgr, 2)
                    label = f"{c}/{TARGET_IMAGES}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img, (x, y - th - 14), (x + tw + 10, y), color_bgr, -1)
                    cv2.putText(img, label, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception:
            return frame

    st.info("📸 **Press START below.** Keep your face in view. The counter on the video will stop at 30/30.")
    
    webrtc_streamer(
        key="register_streamer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": get_ice_servers()}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Complete Registration & Train Model"):
        if len(os.listdir(full_path)) > 0:
            with st.spinner("Processing images — this may take a moment…"):
                try:
                    train_model()
                    st.markdown(f"""
<div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 18px; padding: 2rem; text-align: center; box-shadow: 0 10px 40px rgba(16, 185, 129, 0.3); margin: 1rem 0; border: 3px solid #059669;">
    <h2 style="color: white; margin: 0 0 0.5rem 0; font-size: 1.8rem; font-weight: 700;">Registration Successful!</h2>
    <p style="color: #dcfce7; margin: 0.5rem 0 0; font-size: 1.1rem; font-weight: 500;"><strong>{name}</strong> ({nim}) has been added to the recognition system.</p>
</div>
""", unsafe_allow_html=True)
                except ValueError as e:
                    st.error(f"Cannot complete registration: {e}")
        else:
            st.error("No faces captured. Please ensure the camera sees your face and try again.")
