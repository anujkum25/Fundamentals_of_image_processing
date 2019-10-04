import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread("imdata/doc5.png")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

r = img[:, :, 0]
g = img[:, :, 1]
b = img[:, :, 2]

h = hls[:, :, 0]
l = hls[:, :, 1]
s = hls[:, :, 2]

h1 = hsv[:, :, 0]
s1 = hsv[:, :, 1]
v1 = hsv[:, :, 2]



gray[gray<=0.8] = 0
gray[gray>0.8] = 1

h[h<250] = 0
h[h>=250] = 1


s1[s1<0.2] = 0
s1[s1>=0.2] = 1

def deskew(img):
    m = cv2.moments(img)
    x, y = img.shape
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5*y*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (y, x), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

plt.imshow(h1, cmap="gray")
plt.show()