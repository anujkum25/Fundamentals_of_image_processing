import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('./imdata/onion.png')

# Increase intensity
img = np.multiply(img, 2)
plt.imshow(img)
plt.show()