""" Now use the model created in step 14 to predict """

import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

pika_dir = os.path.join('./tmp/pokemon-test-set/pikachu')
char_dir = os.path.join('./tmp/pokemon-test-set/charizard')

print('total training pikachu images:', len(os.listdir(pika_dir)))
print('total training charizard images:', len(os.listdir(char_dir)))

pika_files = os.listdir(pika_dir)
print(pika_files[:10])

char_files = os.listdir(char_dir)
print(char_files[:10])


flist = pika_files+char_files

var_0 = [pika_files, char_files]
var_1 = [pika_dir, char_dir]


model_path = './pokemon.h5'
model = tf.keras.models.load_model(model_path, custom_objects=None, compile=True, options=None)

i = 0
for _ in var_0:
    for fname in var_0[i]:
        p = os.path.join(var_1[i], fname)

        img = image.load_img(p, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images, batch_size=5)

        if classes[0, 0] == 1.0:
            print(p, classes, 'charizard')

        elif classes[0, 1] == 1.0:
            print(p, classes, 'pikachu')

        # print(p, classes[0, 0])
        # print(p, classes[0, 1])

    i += 1
