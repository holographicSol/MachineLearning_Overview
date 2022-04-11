"""

Exercise 6:
Consider the impact of training for more or less epochs. Why do you think that would be the case?

Try 15 epochs -- you'll probably get a model with a much better loss than the one with 5 Try 30 epochs -- you might see
the loss value stops decreasing, and sometimes increases. This is a side effect of something called 'overfitting' which
you can learn about [somewhere] and it's something you need to keep an eye out for when training neural networks.
There's no point in wasting your time training if you aren't improving your loss, right! :)

"""

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
epochs = [5, 15, 30]
time_all = []

i = 0
for _ in epochs:
    print('-' * 100)

    time_0 = time.time()

    mnist = tf.keras.datasets.fashion_mnist

    (training_images, training_labels),  (test_images, test_labels) = mnist.load_data()

    training_images = training_images/255.0
    test_images = test_images/255.0

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')

    model.fit(training_images, training_labels, epochs=epochs[i])

    model.evaluate(test_images, test_labels)

    classifications = model.predict(test_images)

    time_1 = time.time()
    time_2 = time_1 - time_0
    time_all.append(time_2)

    print('number of epochs:', epochs[i])
    print('classifications:', classifications[34])
    print('test_labels:', test_labels[34])
    print('time taken:', time_2)

    i += 1

