import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
import skimage
from skimage.transform import rescale,resize
import imageio
from scipy import misc
from skimage import data, color
from skimage.transform import rescale, resize
from sklearn.ensemble import ExtraTreesClassifier
from skimage.segmentation import clear_border
from scipy.ndimage import label, generate_binary_structure
from sklearn.cluster import KMeans
import matplotlib.patches as patches

import os
import cv2
from skimage import io

retval = os.getcwd()
print("Current working directory %s" % retval)



data = pd.read_csv("./data/train.csv").as_matrix()
print(data.shape)
clf = ExtraTreesClassifier()
#training dataset

xtrain = data[0:21000,1:]
#print("train dataset",xtrain)
train_label = data[0:21000,0]
#print("train lebel",train_label)
clf.fit(xtrain,train_label)

#testing dataset

xtest = data[21000:,1:]
actual_label = data[21000:,0]

p=clf.predict(xtest)
count=0
for i in range(0,21000):
    count+= 1 if p[i]==actual_label[i] else 0

print("accuracy=", (count*100)/21000)




j=255
d = xtest[j]
print("d",d)
d.shape = (28,28)
pt.imshow(240-d,cmap='gray')
pt.imshow(240-d,cmap='gray')
print("predict",clf.predict([xtest[j]]))
pt.show()


image = imageio.imread("./data/doc5.png")
#image = resize(img,(28,28))
print(image)
#from PIL import Image
i = Image.open('./data/doc5.png')
iar = np.asarray(i)
#print(iar)
#image = cv2.imread("./data/doc5.png")
image1 = cv2.cvtColor(cv2.imread('./data/doc5.png'),
                  cv2.COLOR_BGR2RGB)
#image = resize(image,(28,28))
#i = cv2.imread("./data/doc5.png")
#img1 = cv2.cvtColor(image,cv2.COLOR_RGBA2GRAY)
#pixel = image[95,200]
pixel = image1[106,254]
#+ image[357,317] + image[359,350]
print("pixel",pixel)
image_data = np.asarray(image1)

#img2 = clear_border(img1)
#labeled_image, _ = scipy.ndimage.label(binarized_image)
#for region in np.union1d(labeled_image[[0,-1]].flatten(),
 #                        labeled_image[:,[0,-1]].flatten()):
  #  binarized_image[labeled_image == region] = 0
height, width, dim = image.shape

r = 100.0 / image.shape[1]
dim = (100, int(image.shape[0] * r))
#extracting green image
print(np.shape(image))
for i in range(0,np.shape(image)[0]):
    for j in range(0,np.shape(image)[1]):
        if(image[i,j,1] > image[i,j,0]) & (image[i,j,1] > image[i,j,2]) & (image[i,j,0] < 170):# & (im[i,j,2] < 200):
            image[i,j,0] = 0
            image[i,j,1] = 255
            image[i,j,2] = 0
        else:
            image[i,j,:] = 0


retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image[:,:,1])
print(np.shape(stats))

for i in range(1,np.shape(stats)[0]):
    xstart = stats[i,0]
    xend   = xstart + stats[i,2]
    ystart = stats[i,1]
    yend   = ystart + stats[i,3]

    if(stats[i,3] > 6 ):
        plt.figure(i)
        plt.imshow(image[ystart:yend, xstart:xend], cmap='gray')
        #plt.show()

print(i)
plt.figure(i+1)
plt.imshow(image, cmap='gray')
plt.show()
# perform the actual resizing of the image and show it
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
# crop the image using array slices -- it's a NumPy array
# after all!
#color = ('b','g','r')
#for i,col in enumerate(color):
   #  histr = cv2.calcHist([img],[i],None,[256],[0,256])
     ##plt.plot(histr,color = col)
     #plt.xlim([0,256])
#plt.show()
cropped = image[150:600, 90:200]
cropped1 = cropped[120:400,0:50]
cropped2 = cropped1[120:400,0:20]
io.imshow(image)
cv2.imwrite("./data/train_image_green.png",image)

io.show()
print(image_data[i][j]) 
pt.show()

