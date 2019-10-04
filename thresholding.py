import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('./imdata/onion.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray[gray>128] = 255
gray[gray<=128] = 0

plt.imshow(gray, cmap="gray")
plt.show()