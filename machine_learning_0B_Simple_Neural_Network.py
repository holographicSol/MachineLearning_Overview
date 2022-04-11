import tensorflow as tf
import numpy as np
from tensorflow import keras

"""
Create the simplest possible neural network. It has one layer, that layer has one neuron, and the input shape to
it is only one value. """
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Define and compile the neural network:
model.compile(optimizer='sgd', loss='mean_squared_error')

# Feed some data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
print(xs)
print(ys)

# Train the neural network:
model.fit(xs, ys, epochs=500)

# Use the model:
print(model.predict([10.0]))
