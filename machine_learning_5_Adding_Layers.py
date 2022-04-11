"""

Exercise 5:
Consider the effects of additional layers in the network. What will happen if you add another layer between the one
with 512 and the final layer with 10.

Ans: There isn't a significant impact -- because this is relatively simple data. For far more complex data (including
color images to be classified as flowers that you'll see in the next lesson), extra layers are often necessary.

"""

import tensorflow as tf
import time
print(tf.__version__)

test_labels_all = []
classifications_all = []
layers = [2, 3]
time_all = []

i = 0
for _ in layers:

    time_0 = time.time()

    mnist = tf.keras.datasets.fashion_mnist

    (training_images, training_labels),  (test_images, test_labels) = mnist.load_data()

    training_images = training_images/255.0
    test_images = test_images/255.0

    if i is 0:
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    else:
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                            tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')

    model.fit(training_images, training_labels, epochs=5)

    model.evaluate(test_images, test_labels)

    classifications = model.predict(test_images)

    # data aggregation
    classifications_all.append(classifications)
    test_labels_all.append(test_labels)
    time_1 = time.time()
    time_2 = time_1 - time_0
    time_all.append(time_2)

    i += 1

print('')
i = 0
for _ in time_all:
    print('-' * 100)
    print('number of layers:', layers[i])
    print('classifications[0]:', classifications_all[i])
    print('test_labels[0]:', test_labels_all[i])
    print('time taken:', time_all[i])

    i += 1
