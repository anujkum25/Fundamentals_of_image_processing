import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


c = 1
gamma = 0.5
img = cv2.imread("./imdata/pout.tif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
a = 0
b = 255
c = np.min(img[:, :, 1])
d = np.max(img[:, :, 1])

stretched = np.uint8((img - c) * ((b-a) / (d - c)) + a)

xhist = np.arange(256)
yhist = cv2.calcHist([stretched], [0], None, [256], [0, 256])
eqhist = cv2.equalizeHist(gray)

plt.subplot(141)
plt.imshow(img)
plt.subplot(142)
plt.imshow(stretched)
plt.subplot(143)
# plt.plot(xhist, yhist.ravel())
plt.plot(yhist.ravel())
plt.subplot(144)
plt.imshow(eqhist, cmap="gray")
plt.show()

