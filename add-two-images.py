import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

img1 = cv2.imread('./imdata/onion.png')
img2 = cv2.imread('./imdata/cameraman.tif')

x, y, z = img1.shape

img2[:x, :y, :] = img2[:x, :y, :] + img1

# Add two image
plt.imshow(img2, cmap='gray')
plt.show()
