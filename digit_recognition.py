import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(x_train, y_train, epochs=10)

# Test accuracy
model.evaluate(x_test, y_test)

# Save the trained model
model.save("digit_model.h5")

# Predict example
prediction = model.predict(x_test)

plt.imshow(x_test[0])
plt.show()

print("Predicted Digit:", prediction[0].argmax())