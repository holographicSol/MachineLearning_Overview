"""

Convolutional networks


Start by importing some Python libraries and the ascent picture:

"""

import cv2
import numpy as np
from scipy import misc

i = misc.ascent()

""" Next, use the Pyplot library matplotlib to draw the image so that you know what it looks like: """

import matplotlib.pyplot as plt

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()

"""

You can see that it's an image of a stairwell. There are lots of features you can try and isolate. For example, there
are strong vertical lines.

The image is stored as a NumPy array, so we can create the transformed image by just copying that array. The size_x
and size_y variables will hold the dimensions of the image so you can loop over it later.

"""

i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

print('original size_x:', size_x)
print('original size_y:', size_y)

""" This filter detects edges nicely
It creates a filter that only passes through sharp edges and straight lines. 
Experiment with different values for fun effects. """

"""

Consider the following filter values and their impact on the image.

Using [-1,0,1,-2,0,2,-1,0,1] gives you a very strong set of vertical lines.

Using [-1,-2,-1,0,0,0,1,2,1] gives you horizontal lines.

Explore different values! Also, try differently sized filters, such as 5x5 or 7x7.

"""

# original filter
# filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]

# very strong set of vertical lines
# filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# horizontal lines
# filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

# different sized filter
filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1], [-1, -2, -1], [0, 0, 0]]


""" If all the digits in the filter don't add up to 0 or 1, you
should probably do a weight to get it to do so
so, for example, if your weights are 1,1,1 1,2,1 1,1,1
They add up to 10, so you would set a weight of .1 if you want to normalize them """

weight = 1

"""

Now, calculate the output pixels. Iterate over the image, leaving a 1-pixel margin, and multiply each of the neighbors
of the current pixel by the value defined in the filter.

That means that the current pixel's neighbor above it and to the left of it will be multiplied by the top-left item in
the filter. Then, multiply the result by the weight and ensure that the result is in the range 0 through 255.

Finally, load the new value into the transformed image:

"""

for x in range(1, size_x-1):
    for y in range(1, size_y-1):
        output_pixel = 0.0
        output_pixel = output_pixel + (i[x - 1, y-1] * filter[0][0])
        output_pixel = output_pixel + (i[x, y-1] * filter[0][1])
        output_pixel = output_pixel + (i[x + 1, y-1] * filter[0][2])
        output_pixel = output_pixel + (i[x-1, y] * filter[1][0])
        output_pixel = output_pixel + (i[x, y] * filter[1][1])
        output_pixel = output_pixel + (i[x+1, y] * filter[1][2])
        output_pixel = output_pixel + (i[x-1, y+1] * filter[2][0])
        output_pixel = output_pixel + (i[x, y+1] * filter[2][1])
        output_pixel = output_pixel + (i[x+1, y+1] * filter[2][2])
        output_pixel = output_pixel * weight
        if output_pixel < 0:
            output_pixel = 0
        if output_pixel > 255:
            output_pixel = 255
        i_transformed[x, y] = output_pixel
        # print(i_transformed)

"""

Now, plot the image to see the effect of passing the filter over it:

Note the size of the axes -- they are 512 by 512

"""

size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]
print('size_x:', size_x)
print('size_y:', size_y)

plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
# plt.axis('off')
plt.show()
