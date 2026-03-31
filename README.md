# ✏️ Handwritten Digit Recognition

A deep learning web app that recognizes handwritten digits (0–9) in real time. Draw a digit on the canvas and the CNN model predicts it instantly.

🚀 **Live Demo** → [Click here](https://handwritten-digit-recognition-ai-9mmvdmxqdhbnp3kkh5roke.streamlit.app/)

---

## 📸 Features

- 🖊️ Draw any digit (0–9) on an interactive canvas
- ⚡ Real-time prediction with confidence score
- 📊 Probability bar chart for all 10 digits
- 📈 Training accuracy curve visualization
- 🧠 Model stats displayed in sidebar (accuracy, loss, parameters)

---

## 🧠 Model Architecture

Built with **TensorFlow / Keras**, trained on the **MNIST dataset** (60,000 training images).

```
Input (28×28×1)
    ↓
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPooling → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPooling → Dropout(0.25)
    ↓
Flatten → Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(10) → Softmax
```

| Metric | Value |
|---|---|
| Test Accuracy | ~99% |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Early Stopping | Yes (patience=3) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML Framework | TensorFlow / Keras |
| Drawing Canvas | streamlit-drawable-canvas |
| Charts | Plotly |
| Deployment | Streamlit Cloud |

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/uniquehp19-cb/handwritten-digit-recognition-ai.git
cd handwritten-digit-recognition-ai
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Train the model** (only needed once)
```bash
python train_model.py
```

**5. Launch the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
handwritten-digit-recognition-ai/
├── app.py                 # Streamlit web app
├── train_model.py         # CNN training script
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python version for Streamlit Cloud
├── mnist_cnn.keras        # Trained model
└── model_stats.json       # Accuracy & training history
```

---

## 👤 Author

**Riya**
- 3rd Year AI/ML Engineering Student
- 🏆 First Place — Prompt Prodigy, JJ Engineering College

---

⭐ Star this repo if you found it useful!
