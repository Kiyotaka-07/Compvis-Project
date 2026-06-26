import streamlit as st

st.set_page_config(
    page_title="FaceAttend — Smart Attendance",
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
.block-container { padding-top: 2.5rem; padding-bottom: 10rem; max-width: 780px; }

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
    margin-bottom: 0.85rem;
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
    margin-bottom: 2.2rem;
}

.face-scan-wrapper {
    display: flex;
    justify-content: center;
    margin: 1.2rem 0 2.2rem;
}
.face-ring {
    position: relative;
    width: 120px;
    height: 120px;
}
.face-ring svg {
    width: 120px;
    height: 120px;
    animation: spin-slow 8s linear infinite;
}
@keyframes spin-slow { to { transform: rotate(360deg); } }
.face-icon {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.6rem;
}

.card-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1.1rem;
    margin-bottom: 2rem;
}
@media (max-width: 520px) { .card-grid { grid-template-columns: 1fr; } }

.nav-card {
    background: white;
    border-radius: 18px;
    padding: 1.6rem 1.5rem;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 16px rgba(99,102,241,0.06);
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    text-decoration: none;
    display: block;
}
.nav-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(99,102,241,0.13);
}

.card-icon {
    width: 46px;
    height: 46px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    margin-bottom: 1rem;
}
.card-icon.purple { background: rgba(99,102,241,0.1); }
.card-icon.green  { background: rgba(16,185,129,0.1); }

.card-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1e1b4b;
    margin-bottom: 0.35rem;
}
.card-desc {
    font-size: 0.84rem;
    color: #94a3b8;
    line-height: 1.55;
    margin-bottom: 1rem;
}
.card-arrow {
    font-size: 0.82rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.card-arrow.purple { color: #6366f1; }
.card-arrow.green  { color: #10b981; }

.stats-strip {
    display: flex;
    gap: 0;
    background: white;
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.06);
    overflow: hidden;
    margin-bottom: 0;  /* Change from 2rem to 0 */
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    width: 100%;
}
.stat-item {
    flex: 1;
    padding: 1rem;
    text-align: center;
    border-right: 1px solid rgba(0,0,0,0.05);
}
.stat-item:last-child { border-right: none; }
.stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #1e1b4b;
}
.stat-lbl {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
}

.soft-footer {
    text-align: center;
    color: #cbd5e1;
    font-size: 0.78rem;
    padding-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 class="hero-title">Smart Attendance<br>with <span>Face Detection</span></h1>
<p class="hero-sub">
    Automatically record student attendance using facial recognition.
    Fast, contactless, and accurate — no more paper lists.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="face-scan-wrapper">
  <div class="face-ring">
    <svg viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="60" cy="60" r="54" stroke="#e0e7ff" stroke-width="3"/>
      <circle cx="60" cy="60" r="54" stroke="#6366f1" stroke-width="3"
        stroke-dasharray="60 280" stroke-linecap="round"/>
      <circle cx="60" cy="60" r="42" stroke="#f0fdf4" stroke-width="1.5"/>
      <circle cx="60" cy="60" r="42" stroke="#10b981" stroke-width="1.5"
        stroke-dasharray="30 240" stroke-linecap="round"/>
    </svg>
    <div class="face-icon">🪪</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card-grid">

  <a class="nav-card" href="/Attendance" target="_self">
    <div class="card-icon purple">📋</div>
    <div class="card-title">Take Attendance</div>
    <div class="card-desc">
      Launch the camera and let the system automatically detect and log student faces in real time.
    </div>
    <span class="card-arrow purple">Open &rarr;</span>
  </a>

  <a class="nav-card" href="/Register" target="_self">
    <div class="card-icon green">👤</div>
    <div class="card-title">Register Student</div>
    <div class="card-desc">
      Add a new student by capturing their face and saving their profile to the recognition model.
    </div>
    <span class="card-arrow green">Open &rarr;</span>
  </a>

  <a class="nav-card" href="/Registration_List" target="_self">
    <div class="card-icon purple">🗂️</div>
    <div class="card-title">Manage Dataset</div>
    <div class="card-desc">
      View and manage registered students in the dataset. Delete entries and retrain the model.
    </div>
    <span class="card-arrow purple">Open &rarr;</span>
  </a>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-strip">
  <div class="stat-item">
    <div class="stat-num">Real-time</div>
    <div class="stat-lbl">Detection</div>
  </div>
  <div class="stat-item">
    <div class="stat-num">Auto</div>
    <div class="stat-lbl">Logging</div>
  </div>
</div>
""", unsafe_allow_html=True)
