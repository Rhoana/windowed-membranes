import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

directory_input = 'train-input'
files_input = glob.glob(directory_input+"/*.tif")
directory_labels = 'train-labels'
files_labels = glob.glob(directory_labels+"/*.tif")

n = 99
img = cv2.imread(files_labels[n],cv2.IMREAD_UNCHANGED)      
img_real = cv2.imread(files_input[n],cv2.IMREAD_UNCHANGED)      
print np.unique(img)

value1 = 100
value2 = 300

threshold = 399
print 'threshold = ',threshold

for n in xrange(img.shape[0]):
    for m in xrange(img.shape[1]):
        if img[n,m] > threshold:
            img[n,m] = 1
        elif img[n,m] == 0:
            img[n,m] = 0
        else:
            img[n,m] = 0

img_real += 1
rows,cols = img_real.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,40],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img_real,M,(cols,rows))

plt.subplot(121),plt.imshow(img_real,cmap=plt.cm.gray),plt.title('Input')
plt.subplot(122),plt.imshow(dst,cmap=plt.cm.gray),plt.title('Output')
plt.show()
exit()

plt.figure(1)
plt.imshow(img_real,cmap=plt.cm.gray)
plt.figure(2)
plt.imshow(img,cmap = plt.cm.gray)
plt.show()
