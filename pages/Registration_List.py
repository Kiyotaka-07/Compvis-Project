import streamlit as st
import os
import shutil
import sys

# ── make root importable so train_model() can be found ──────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train_mode import train_model          # adjust if the function name differs

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manage Dataset — FaceAttend",
    page_icon="🗂️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── shared CSS (same design language as Attendance.py) ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

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
    margin-bottom: 0.5rem;
    font-weight: 400;
}
.hero-title span { font-style: italic; color: #6366f1; }
.hero-sub {
    color: #64748b;
    font-size: 1rem;
    line-height: 1.65;
    max-width: 520px;
    margin-bottom: 1.8rem;
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

/* ── person cards ── */
.person-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: white;
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.55rem;
    transition: box-shadow 0.18s;
}
.person-card:hover { box-shadow: 0 4px 18px rgba(99,102,241,0.1); }
.person-card.selected {
    border: 1.5px solid #6366f1;
    background: rgba(99,102,241,0.04);
}
.person-info { display: flex; align-items: center; gap: 0.9rem; }
.avatar {
    width: 40px; height: 40px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #818cf8);
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 1rem;
    flex-shrink: 0;
}
.person-name {
    font-weight: 600;
    color: #1e1b4b;
    font-size: 0.95rem;
}
.person-meta {
    font-size: 0.76rem;
    color: #94a3b8;
    margin-top: 1px;
}

/* ── stats strip ── */
.stats-strip {
    display: flex;
    background: white;
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.06);
    overflow: hidden;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.stat-item { flex: 1; padding: 0.9rem 1rem; text-align: center; border-right: 1px solid rgba(0,0,0,0.05); }
.stat-item:last-child { border-right: none; }
.stat-num { font-family: 'DM Serif Display', serif; font-size: 1.45rem; color: #1e1b4b; }
.stat-lbl { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

/* ── action buttons ── */
div.stButton > button {
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    border: none !important;
    width: 100%;
}

.empty-state {
    background: white;
    border-radius: 14px;
    border: 2px dashed #e2e8f0;
    padding: 2.5rem;
    text-align: center;
    color: #94a3b8;
}
.empty-state-icon { font-size: 2.5rem; margin-bottom: 0.5rem; opacity: 0.5; }

div[data-testid="stAlert"] {
    display: none !important;
}
            
.custom-success {
    display: flex;
    align-items: center;
    gap: 14px;
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border: 1.5px solid #22c55e;
    border-left: 6px solid #16a34a;
    border-radius: 14px;
    padding: 1rem 1.15rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 22px rgba(34, 197, 94, 0.18);
}

.custom-success-icon {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: #16a34a;
    color: white !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    font-weight: 800;
    flex-shrink: 0;
}

.custom-success-text {
    color: #14532d !important;
}

.custom-success-title {
    color: #14532d !important;
    font-size: 0.98rem;
    font-weight: 800;
    margin-bottom: 2px;
}

.custom-success-subtitle {
    color: #166534 !important;
    font-size: 0.82rem;
    font-weight: 500;
    opacity: 0.95;
}
</style>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(ROOT, "dataset")

def get_persons():
    """Return list of (display_name, folder_name, image_count) from dataset/."""
    if not os.path.isdir(DATASET_DIR):
        return []
    persons = []
    for folder in sorted(os.listdir(DATASET_DIR)):
        full = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(full):
            continue
        # folder format is  Name_ID  — extract the display name
        display = folder.rsplit("_", 1)[0].replace("_", " ")
        imgs = [f for f in os.listdir(full) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        persons.append((display, folder, len(imgs)))
    return persons

# ── session state ─────────────────────────────────────────────────────────────
if "selected" not in st.session_state:
    st.session_state.selected = set()

# ── header ────────────────────────────────────────────────────────────────────
st.markdown('<a class="back-link" href="/" target="_self">← Back to Dashboard</a>', unsafe_allow_html=True)
st.markdown("""
<h1 class="hero-title">Manage <span>Dataset</span></h1>
<p class="hero-sub">
    Select one or more people to remove from the dataset, then retrain the
    model so the changes take effect immediately.
</p>
""", unsafe_allow_html=True)



# ── fetch persons ─────────────────────────────────────────────────────────────
persons = get_persons()
total_imgs = sum(c for _, _, c in persons)

# ── stats strip ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stats-strip">
  <div class="stat-item">
    <div class="stat-num">{len(persons)}</div>
    <div class="stat-lbl">People</div>
  </div>
  <div class="stat-item">
    <div class="stat-num">{total_imgs}</div>
    <div class="stat-lbl">Total Images</div>
  </div>
  <div class="stat-item">
    <div class="stat-num">{len(st.session_state.selected)}</div>
    <div class="stat-lbl">Selected</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── person list ───────────────────────────────────────────────────────────────
if not persons:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">📂</div>
        <div style="font-weight:600;color:#475569;">No entries found</div>
        <div style="font-size:0.82rem;margin-top:4px;">Add people via the Register page first.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    all_folders = {folder for _, folder, _ in persons}

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

    for display, folder, img_count in persons:
        initials = "".join(w[0].upper() for w in display.split()[:2])
        is_sel = folder in st.session_state.selected
        card_cls = "person-card selected" if is_sel else "person-card"
        col_card, col_check = st.columns([9, 1])

        with col_card:
            st.markdown(f"""
            <div class="{card_cls}">
              <div class="person-info">
                <div class="avatar">{initials}</div>
                <div>
                  <div class="person-name">{display}</div>
                  <div class="person-meta">{img_count} image{"s" if img_count != 1 else ""} · {folder}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_check:
            # vertical centering trick
            st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
            def toggle_selection(folder_id):
                if folder_id in st.session_state.selected:
                    st.session_state.selected.discard(folder_id)
                else:
                    st.session_state.selected.add(folder_id)
            
            st.checkbox("", value=is_sel, key=f"chk_{folder}", on_change=toggle_selection, args=(folder,))

# ── action buttons ────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

n_sel = len(st.session_state.selected)
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    delete_label = f"🗑️  Delete {n_sel} selected" if n_sel else "🗑️  Delete selected"
    delete_clicked = st.button(delete_label, disabled=n_sel == 0)

with btn_col2:
    retrain_clicked = st.button("🔄  Retrain model")

# ── delete flow ───────────────────────────────────────────────────────────────
if delete_clicked and n_sel > 0:
    errors = []
    for folder in list(st.session_state.selected):
        target = os.path.join(DATASET_DIR, folder)
        try:
            shutil.rmtree(target)
        except Exception as e:
            errors.append(f"{folder}: {e}")

    st.session_state.selected = set()
    st.experimental_rerun()

# ── retrain flow ──────────────────────────────────────────────────────────────
if retrain_clicked:
    if len(persons) == 0:
        st.error("No data in dataset — add people first.")
    else:
        with st.spinner("Training model… this may take a moment."):
            try:
                train_model()
                st.markdown("""
<div class="custom-success">
    <div class="custom-success-icon">✓</div>
    <div class="custom-success-text">
        <div class="custom-success-title">Model retrained successfully!</div>
        <div class="custom-success-subtitle">Your face recognition model has been updated with the latest dataset.</div>
    </div>
</div>
""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Training failed: {e}")

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<p style="text-align:center;color:#cbd5e1;font-size:0.78rem;padding-top:1.5rem;">'
    "FaceAttend · Dataset Manager</p>",
    unsafe_allow_html=True,
)