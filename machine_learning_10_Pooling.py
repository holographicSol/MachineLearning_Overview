"""

Understanding Pooling:
Now that you've identified the essential features of the image, what do you do? How do you use the resulting feature
map to classify images?

Similar to convolutions, pooling greatly helps with detecting features. Pooling layers reduce the overall amount of
information in an image while maintaining the features that are detected as present.

There are a number of different types of pooling, but you'll use one called Maximum (Max) Pooling.

Iterate over the image and, at each point, consider the pixel and its immediate neighbors to the right, beneath, and
right-beneath. Take the largest of those (hence max pooling) and load it into the new image. Thus, the new image will
be one-fourth the size of the old.


Write code for pooling:
The following code will show a (2, 2) pooling. Run it to see the output.

You'll see that while the image is one-fourth the size of the original, it kept all the features.

"""

import cv2
import numpy as np
from scipy import misc

i = misc.ascent()

""" Next, use the Pyplot library matplotlib to draw the image so that you know what it looks like: """

import matplotlib.pyplot as plt

i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

print('original size_x:', size_x)
print('original size_y:', size_y)

new_x = int(size_x / 2)
new_y = int(size_y / 2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels = []
        pixels.append(i_transformed[x, y])
        pixels.append(i_transformed[x + 1, y])
        pixels.append(i_transformed[x, y + 1])
        pixels.append(i_transformed[x + 1, y + 1])
        pixels.sort(reverse=True)
        newImage[int(x / 2), int(y / 2)] = pixels[0]

size_x = newImage.shape[0]
size_y = newImage.shape[1]

print('newImage size_x:', size_x)
print('newImage size_y:', size_y)

# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
# plt.axis('off')
plt.show()

"""

Congratulations
You've built your first computer vision model! To learn how to further enhance your computer vision models, proceed to
build convolutional neural networks (CNNs) to enhance computer vision.

"""