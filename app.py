import streamlit as st
import numpy as np
import json
import os
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="✏️",
    layout="wide",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background: #0d0d0f; color: #e8e6e1; }
  h1, h2, h3 { font-family: 'Space Mono', monospace !important; letter-spacing: -0.03em; }
  .hero-title { font-family: 'Space Mono', monospace; font-size: clamp(2rem, 5vw, 3.5rem); font-weight: 700; color: #f0ede8; line-height: 1.1; margin-bottom: 0.25rem; }
  .hero-sub { font-size: 1rem; color: #6b6b70; margin-bottom: 2rem; font-weight: 300; letter-spacing: 0.02em; }
  .pred-badge { display: inline-flex; align-items: center; justify-content: center; width: 120px; height: 120px; border-radius: 50%; background: #1a1a1f; border: 2px solid #3d3d45; font-family: 'Space Mono', monospace; font-size: 3.5rem; font-weight: 700; color: #c8f542; margin: 0 auto 0.5rem; }
  .conf-text { font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #6b6b70; text-align: center; letter-spacing: 0.05em; }
  .pred-label { font-size: 0.75rem; color: #4a4a52; text-align: center; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.25rem; }
  .stat-card { background: #14141a; border: 1px solid #222228; border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 0.75rem; }
  .stat-label { font-size: 0.7rem; color: #4a4a52; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 4px; }
  .stat-value { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #c8f542; }
  .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
  .bar-digit { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #6b6b70; width: 14px; text-align: right; }
  .bar-bg { flex: 1; height: 10px; background: #1a1a1f; border-radius: 3px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 3px; }
  .bar-pct { font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #4a4a52; width: 42px; text-align: right; }
  .canvas-hint { font-size: 0.78rem; color: #3d3d45; text-align: center; margin-top: 6px; letter-spacing: 0.04em; }
  .stButton > button { background: #c8f542 !important; color: #0d0d0f !important; border: none !important; border-radius: 8px !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; font-size: 0.85rem !important; letter-spacing: 0.06em !important; padding: 0.6rem 1.4rem !important; width: 100% !important; }
  .stButton > button:hover { background: #d4f75c !important; }
  section[data-testid="stSidebar"] { background: #0a0a0c !important; border-right: 1px solid #1c1c22 !important; }
  div[data-testid="stSlider"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists("mnist_cnn.h5"):
        return None
    return tf.keras.models.load_model("mnist_cnn.h5")

@st.cache_data
def load_stats():
    if not os.path.exists("model_stats.json"):
        return None
    with open("model_stats.json") as f:
        return json.load(f)

model = load_model()
stats = load_stats()

def preprocess_canvas(canvas_data):
    # canvas is RGBA with white strokes on dark background
    img = Image.fromarray(canvas_data.astype("uint8"), "RGBA")

    # Extract alpha channel — where user drew has high alpha
    r, g, b, a = img.split()

    # Use the RGB brightness as the digit mask
    # White strokes on dark bg → bright pixels = digit
    gray = img.convert("L")

    # Slightly blur to smooth jagged edges (like MNIST style)
    gray = gray.filter(ImageFilter.GaussianBlur(radius=1))

    # Crop tightly around the digit to center it (MNIST digits are centered)
    bbox = gray.getbbox()
    if bbox:
        gray = gray.crop(bbox)

    # Add padding around the cropped digit
    pad = 20
    padded = Image.new("L", (gray.width + pad*2, gray.height + pad*2), 0)
    padded.paste(gray, (pad, pad))
    gray = padded

    # Resize to 28x28
    gray = gray.resize((28, 28), Image.LANCZOS)

    # Convert to array — white digit on black bg (matches MNIST format)
    arr = np.array(gray).astype("float32") / 255.0

    return arr.reshape(1, 28, 28, 1)

if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0

with st.sidebar:
    st.markdown('<div style="padding:1.5rem 0 1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">✏️<br>Digit<br>Recognizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">CNN trained on MNIST</div>', unsafe_allow_html=True)
    if not model:
        st.error("Model not found. Run `python train_model.py` first.", icon="⚠️")
    else:
        st.markdown('<div style="height:1px;background:#1c1c22;margin:1rem 0;"></div>', unsafe_allow_html=True)
        st.markdown("**Model info**")
        if stats:
            st.markdown(f"""
            <div class="stat-card"><div class="stat-label">Test accuracy</div><div class="stat-value">{stats['test_accuracy']}%</div></div>
            <div class="stat-card"><div class="stat-label">Test loss</div><div class="stat-value">{stats['test_loss']}</div></div>
            <div class="stat-card"><div class="stat-label">Parameters</div><div class="stat-value">{stats['model_params']:,}</div></div>
            <div class="stat-card"><div class="stat-label">Epochs trained</div><div class="stat-value">{stats['train_epochs']}</div></div>
            """, unsafe_allow_html=True)
        st.markdown('<div style="height:1px;background:#1c1c22;margin:1rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.75rem;color:#3d3d45;line-height:1.7;">Architecture<br><span style="color:#5a5a62;">Conv2D x4 → MaxPool → Dropout → Dense</span></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if not model:
    st.markdown("## Setup required\n\nRun `python train_model.py` first.")
    st.stop()

col_canvas, col_result = st.columns([1.1, 0.9], gap="large")

with col_canvas:
    st.markdown('<div style="padding-top:1rem;">', unsafe_allow_html=True)
    st.markdown("#### Draw a digit")
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=18,
        stroke_color="#ffffff",
        background_color="#111115",
        height=340,
        width=340,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state['canvas_key']}",
        display_toolbar=False,
    )
    st.markdown('<div class="canvas-hint">draw clearly · center your digit · use thick strokes</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        predict_btn = st.button("PREDICT →")
    with col_b:
        clear_btn = st.button("CLEAR")
    if clear_btn:
        st.session_state["canvas_key"] += 1
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col_result:
    st.markdown('<div style="padding-top:1rem;">', unsafe_allow_html=True)
    st.markdown("#### Prediction")
    prediction = None
    confidence = None
    all_probs  = None
    has_drawing = (canvas_result.image_data is not None and canvas_result.image_data[..., :3].sum() > 500)
    if predict_btn and has_drawing:
        tensor     = preprocess_canvas(canvas_result.image_data)
        probs      = model.predict(tensor, verbose=0)[0]
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction]) * 100
        all_probs  = probs
    elif predict_btn and not has_drawing:
        st.warning("Draw a digit on the canvas first.")
    if prediction is not None:
        st.markdown(f"""
        <div style="text-align:center;padding:1.5rem 0 1rem;">
          <div class="pred-label">predicted digit</div>
          <div class="pred-badge">{prediction}</div>
          <div class="conf-text">{confidence:.1f}% confidence</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div style="margin-top:1rem;"><div style="font-size:0.7rem;color:#4a4a52;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">All digit probabilities</div>', unsafe_allow_html=True)
        bars_html = ""
        for i, p in enumerate(all_probs):
            pct   = p * 100
            color = "#c8f542" if i == prediction else "#2a2a32"
            bars_html += f'<div class="bar-row"><div class="bar-digit">{i}</div><div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div></div><div class="bar-pct">{pct:.1f}%</div></div>'
        st.markdown(bars_html + "</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 0;color:#2a2a32;">
          <div style="font-family:'Space Mono',monospace;font-size:5rem;margin-bottom:1rem;">?</div>
          <div style="font-size:0.8rem;letter-spacing:0.1em;text-transform:uppercase;">awaiting input</div>
        </div>""", unsafe_allow_html=True)
    if stats and stats.get("train_acc_history"):
        st.markdown('<div style="height:1px;background:#1c1c22;margin:1.5rem 0 1rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.7rem;color:#4a4a52;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">Training accuracy curve</div>', unsafe_allow_html=True)
        import plotly.graph_objects as go
        fig = go.Figure()
        ep = list(range(1, len(stats["train_acc_history"]) + 1))
        fig.add_trace(go.Scatter(x=ep, y=[v*100 for v in stats["val_acc_history"]], name="Validation", line=dict(color="#c8f542", width=2), fill="tozeroy", fillcolor="rgba(200,245,66,0.06)"))
        fig.add_trace(go.Scatter(x=ep, y=[v*100 for v in stats["train_acc_history"]], name="Train", line=dict(color="#3d3d45", width=1.5, dash="dot")))
        fig.update_layout(height=180, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0), legend=dict(font=dict(color="#4a4a52",size=10),bgcolor="rgba(0,0,0,0)"), xaxis=dict(showgrid=False,color="#3d3d45",title="Epoch",title_font_size=10), yaxis=dict(showgrid=True,gridcolor="#1c1c22",color="#3d3d45",title="Accuracy %",title_font_size=10,range=[90,100]))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)