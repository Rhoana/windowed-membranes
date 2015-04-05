import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

directory_input = 'train-input'
files_input = glob.glob(directory_input+"/*.tif")
directory_labels = 'train-labels'
files_labels = glob.glob(directory_labels+"/*.tif")

img = cv2.imread(files_labels[-1],cv2.IMREAD_UNCHANGED)      
img_real = cv2.imread(files_input[-1],cv2.IMREAD_UNCHANGED)      
print np.unique(img)

value1 = 100
value2 = 300

threshold = 400

for n in xrange(img.shape[0]):
    for m in xrange(img.shape[1]):
        if img[n,m] > threshold:
            img[n,m] = value1
        elif img[n,m] == 0:
            img[n,m] = 0
        else:
            img[n,m] = value2

plt.figure(1)
plt.imshow(img_real,cmap=plt.cm.gray)
plt.figure(2)
plt.imshow(img)
plt.show()
