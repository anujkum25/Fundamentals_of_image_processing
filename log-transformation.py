import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab as p

c = 1
img = cv2.imread("./imdata/mri.tif")
log = c * np.log(np.add(img, [1, 1, 1]))

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(log)

plt.show()
p.show()