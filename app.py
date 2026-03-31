import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import onnxruntime as ort

# -----------------------------
# Load the ONNX model
# -----------------------------
ort_session = ort.InferenceSession("digit_model.onnx")

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("Handwritten Digit Recognition")

st.write("Draw a digit (0-9) below:")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",  # Black color
    stroke_width=20,
    stroke_color="#FFFFFF",  # White stroke for drawing
    background_color="#000000",  # Black background
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# When the user draws something
if canvas_result.image_data is not None:
    # Convert canvas to grayscale and resize to 28x28
    img = Image.fromarray(canvas_result.image_data.astype(np.uint8)).convert("L")
    img = ImageOps.invert(img)  # Invert colors: background black -> white
    img = img.resize((28, 28))
    
    # Show what the user drew
    st.image(img, caption="Processed Image", width=140)

    # Convert image to numpy array
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize 0-1
    img_array = img_array.reshape(1, 28, 28, 1).astype(np.float32)

    # -----------------------------
    # Run ONNX inference
    # -----------------------------
    outputs = ort_session.run(None, {"input": img_array})
    pred = outputs[0]
    predicted_digit = np.argmax(pred)

    # Show result
    st.subheader(f"Predicted Digit: {predicted_digit}")
