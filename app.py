# app.py
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import onnxruntime as ort

# -----------------------------
# Load the ONNX model
# -----------------------------
# Make sure digit_model.onnx is in the same folder
ort_session = ort.InferenceSession("digit_model.onnx")

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("Handwritten Digit Recognition (ONNX)")

st.write("Draw a digit (0-9) below:")

# Create a drawable canvas
canvas_result = st_canvas(
    fill_color="black",  # Drawing color
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -----------------------------
# Prediction logic
# -----------------------------
if canvas_result.image_data is not None:
    # Convert the drawn image to grayscale 28x28
    img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img)
    
    # Normalize and reshape for model
    img_array = img_array / 255.0  # scale to 0-1
    img_array = img_array.reshape(1, 28, 28, 1).astype(np.float32)
    
    # Run ONNX inference
    outputs = ort_session.run(None, {"input": img_array})
    pred = np.argmax(outputs[0])
    
    st.write(f"**Predicted Digit: {pred}**")

# -----------------------------
# Optional: Clear canvas button
# -----------------------------
if st.button("Clear Canvas"):
    st.experimental_rerun()
