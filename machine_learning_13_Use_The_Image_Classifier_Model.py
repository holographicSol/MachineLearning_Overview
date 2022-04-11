import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

rock_dir = os.path.join('./tmp/rps/rock')
paper_dir = os.path.join('./tmp/rps/paper')
scissors_dir = os.path.join('./tmp/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

flist = rock_files+paper_files+scissors_files

var_0 = [rock_files, paper_files, scissors_files]
var_1 = [rock_dir, paper_dir, scissors_dir]


model_path = './rps.h5'
model = tf.keras.models.load_model(model_path, custom_objects=None, compile=True, options=None)

i = 0
for _ in var_0:
    for fname in var_0[i]:
        p = os.path.join(var_1[i], fname)

        img = image.load_img(p, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        print(p, classes)

    i += 1
