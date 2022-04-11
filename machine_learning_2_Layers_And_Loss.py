"""

Exercise 2:
Let's now look at the layers in your model. Experiment with different values for the dense layer with 512 neurons. What
different results do you get for loss, training time etc? Why do you think that's the case?

"""
import datetime, time

import tensorflow as tf
print(tf.__version__)

test_labels_all = []
classifications_all = []
neurons = [512, 1024]
time_all = []

for _ in neurons:

    time_0 = time.time()

    mnist = tf.keras.datasets.fashion_mnist

    (training_images, training_labels),  (test_images, test_labels) = mnist.load_data()

    training_images = training_images/255.0
    test_images = test_images/255.0

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(_, activation=tf.nn.relu),
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

print('')
i = 0
for _ in time_all:
    print('-' * 100)
    print('number of neurons:', neurons[i])
    print('classifications[0]:', classifications_all[i])
    print('test_labels[0]:', test_labels_all[i])
    print('time taken:', time_all[i])

    i += 1

"""

adding more Neurons we have to do more calculations, slowing down the process, but in this case they have a good impact
-- we do get more accurate. That doesn't mean it's always a case of 'more is better', you can hit the law of
diminishing returns very quickly!

"""
