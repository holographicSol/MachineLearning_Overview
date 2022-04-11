"""

Exercise 7:
Before you trained, you normalized the data, going from values that were 0-255 to values that were 0-1. What would be
the impact of removing that? Here's the complete code to give it a try. Why do you think you get different results?

"""

import tensorflow as tf
print(tf.__version__)

i = 0


def funk():
    global i
    print('-' * 100)

    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    # To experiment with removing normalization, comment out the following 2 lines
    if i is 0:
        training_images = training_images/255.0
        test_images = test_images/255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(training_images, training_labels, epochs=5)
    model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)

    print('classifications:', classifications[0])
    print('test_labels:', test_labels[0])

    i += 1


funk()
funk()
