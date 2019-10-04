import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import numpy as np
from scipy import signal
from scipy import misc
from skimage import io, viewer


img = mpimg.imread("imdata/cameraman.tif")

#  If read using cv2
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# If read using mpimg
gray = img

# Averaging
mask1 = np.ones((3, 3))*1/9

# Laplacian averaging mask
mask2 = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
mask3 = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])
mask4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])


convolved1 = signal.convolve2d(gray, mask1, mode='same', boundary='fill', fillvalue=0)
convolved2 = signal.convolve2d(gray, mask2, mode='same', boundary='fill', fillvalue=0)
convolved3 = signal.convolve2d(gray, mask3, mode='same', boundary='fill', fillvalue=0)
convolved4 = signal.convolve2d(gray, mask4, mode='same', boundary='fill', fillvalue=0)
# convolved4 = cv2.filter2D(gray, -1,  mask4)
# convolved[convolved>0.4] = 1
# convolved[convolved<0.4] = 0


plt.subplot(151)
plt.imshow(img, cmap="gray")
plt.subplot(152)
plt.imshow(convolved1, cmap="gray")
plt.subplot(153)
# High boost filtering
# hb = (2*img - convolved1)
# hb  = 255 * ((hb-np.min(hb))/(np.max(hb)-np.min(hb)))
# plt.imshow(hb, cmap="gray")
plt.imshow(convolved2, cmap="gray")
plt.subplot(154)
plt.imshow(convolved3, cmap="gray")
plt.subplot(155)
plt.imshow(convolved4, cmap="gray")

plt.show()