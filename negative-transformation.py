import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


img = cv2.imread("./imdata/cameraman.tif")
negative = np.abs(np.subtract(img, [255, 255, 255]))

plt.imshow(negative)

plt.show()
