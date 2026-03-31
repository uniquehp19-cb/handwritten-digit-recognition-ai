# app.py

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import onnxruntime as ort

# -----------------------------
# Load the ONNX model
# -----------------------------
ort_session = ort.InferenceSession("digit_model.onnx")  # Ensure this file exists

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered")
st.title("✍️ Handwritten Digit Recognition")

st.write("Draw a digit (0-9) below:")

# Create a drawable canvas
canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=12,
    stroke_color="#FFFFFF",  # White pen
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict") and canvas_result.image_data is not None:
    # Get image from canvas
    img = Image.fromarray(np.uint8(canvas_result.image_data))
    
    # Convert to grayscale
    img = ImageOps.grayscale(img)
    
    # Resize to 28x28 (MNIST size)
    img = img.resize((28, 28))
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch and channel dimension (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1).astype(np.float32)
    
    # ONNX inference
    outputs = ort_session.run(None, {"input": img_array})
    
    # Get predicted digit
    pred = np.argmax(outputs[0], axis=1)[0]
    
    st.success(f"Predicted Digit: **{pred}**")
