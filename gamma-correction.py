import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


c = 1
gamma = 0.5
img = cv2.imread("./imdata/cameraman.tif")

img1 = img/255.0

gamma_corrected = cv2.pow(img1, gamma)


gamma_corrected = np.uint8(gamma_corrected*255)

print(gamma_corrected)

plt.imshow(gamma_corrected)

plt.show()
