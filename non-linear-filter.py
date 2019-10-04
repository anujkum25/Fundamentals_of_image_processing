import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import numpy as np

# Low pass filter

img = cv2.imread("imdata/cameraman.tif")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def noise(img):
    r, c = img.shape
    for i in range(r):
        for j in range(c):
            img[i][j] = 0 if np.random.randint(100, size=1)[0] == 0 else 255 if np.random.randint(100, size=1)[0] == 1 else img[i][j]
    return img


def non_linear_filter(img, kernel_size):
    r, c = img.shape
    cp = img.copy()
    for i in range(r-kernel_size):
        for j in range(c-kernel_size):
            values = img[i:i+kernel_size, j:j+kernel_size].flatten()
            values = np.sort(values)
            median = values[kernel_size**2 // 2]
            cp[i+kernel_size//2][j+kernel_size//2] = median
    return cp


noisy = noise(gray)
filtered = non_linear_filter(noisy, kernel_size=3)

plt.subplot(131)
plt.imshow(img, cmap="gray")
plt.subplot(132)
plt.imshow(noisy, cmap="gray")
plt.subplot(133)
plt.imshow(filtered, cmap="gray")

plt.show()