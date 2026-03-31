import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model  # only for loading model
from streamlit_drawable_canvas import st_canvas

# Load pre-trained model
model = load_model("digit_model.h5")

st.title("Handwritten Digit Recognition")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Preprocess the drawn image
    img = Image.fromarray(np.uint8(canvas_result.image_data[:, :, 0]))
    img = img.resize((28, 28)).convert('L')
    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img_array)
    st.write("Predicted Digit:", prediction.argmax())
