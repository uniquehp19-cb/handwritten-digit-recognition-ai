import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import json
import os

print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]  # (60000, 28, 28, 1)
x_test  = x_test[..., np.newaxis]   # (10000, 28, 28, 1)

print(f"Train samples : {len(x_train)}")
print(f"Test  samples : {len(x_test)}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── Train ──────────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint("mnist_cnn.h5", save_best_only=True, verbose=1),
]

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluate ───────────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy : {test_acc * 100:.2f}%")
print(f"Test loss     : {test_loss:.4f}")

# Save stats for the Streamlit app to read
stats = {
    "test_accuracy": round(float(test_acc) * 100, 2),
    "test_loss": round(float(test_loss), 4),
    "train_epochs": len(history.history["accuracy"]),
    "train_acc_history": [round(v, 4) for v in history.history["accuracy"]],
    "val_acc_history":   [round(v, 4) for v in history.history["val_accuracy"]],
    "model_params": model.count_params(),
}

with open("model_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\nSaved: mnist_cnn.h5  |  model_stats.json")
print("Run `streamlit run app.py` to launch the app.")
