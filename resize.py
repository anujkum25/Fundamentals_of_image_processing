import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

img = cv2.imread("./imdata/toysnoflash.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img1 = cv2.resize(gray, (1024, 1024))
img2 = cv2.resize(img1, (512, 512))
img3 = cv2.resize(img1, (256, 256))
img4 = cv2.resize(img1, (128, 128))
img5 = cv2.resize(img1, (64, 64))

plt.subplot(231)
plt.imshow(gray)
plt.subplot(232)
plt.imshow(img1)
plt.subplot(233)
plt.imshow(img2)
plt.subplot(234)
plt.imshow(img3)
plt.subplot(235)
plt.imshow(img4)
plt.subplot(236)
plt.imshow(img5)
plt.show()
