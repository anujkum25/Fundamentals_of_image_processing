import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.image as mpimg

img = mpimg.imread('./imdata/onion.png')


img[50:60, 50:60, 0] = 255
img[50:60, 50:60, 1] = 0
img[50:60, 50:60, 2] = 0
plt.imshow(img)
plt.show()