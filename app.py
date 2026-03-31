import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")

st.title("✍️ Handwritten Digit Recognition AI")

st.write("Draw a number (0–9) below and click Predict")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Predict button
if st.button("Predict Digit"):
    if canvas_result.image_data is not None:

        img = Image.fromarray(canvas_result.image_data.astype('uint8'))

        img = img.convert('L')  # convert to grayscale
        img = img.resize((28,28))

        img = np.array(img)
        img = img / 255.0
        img = img.reshape(1,28,28)

        prediction = model.predict(img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Predicted Digit: {digit}")
        st.info(f"Confidence: {confidence*100:.2f}%")

# Clear button
if st.button("Clear Canvas"):
    st.rerun()