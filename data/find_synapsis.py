import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

affine = False

directory_input = 'train-input'
files_input = sorted(glob.glob(directory_input+"/*.tif"))
directory_labels = 'train-labels'
files_labels = sorted(glob.glob(directory_labels+"/*.tif"))

n = -1
img = cv2.imread(files_labels[n],cv2.IMREAD_UNCHANGED)      
img_real = cv2.imread(files_input[n],cv2.IMREAD_UNCHANGED)      

value1 = 100
value2 = 300

threshold = 0
print 'threshold = ',threshold

#for n in xrange(img.shape[0]):
#    for m in xrange(img.shape[1]):
#        if img[n,m] > threshold:
#            img[n,m] = 1
#        elif img[n,m] == 0:
#            img[n,m] = 0
#        else:
#            img[n,m] = 0

plt.figure()
plt.imshow(img_real)

plt.figure()
plt.imshow(img)
plt.show()
exit()

if affine == True:
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,40],[200,50],[100,250]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img_real,M,(cols,rows))

    plt.subplot(121),plt.imshow(img_real,cmap=plt.cm.gray),plt.title('Input')
    plt.subplot(122),plt.imshow(dst,cmap=plt.cm.gray),plt.title('Output')

    dst = cv2.warpAffine(img,M,(cols,rows))
    plt.subplot(121),plt.imshow(img,cmap=plt.cm.jet,alpha=0.2),plt.title('Input')
    plt.subplot(122),plt.imshow(dst,cmap=plt.cm.jet,alpha=0.2),plt.title('Output')
    plt.show()
    exit()

x_synapse,y_synapse = np.where(img == 1)

norm_img_real = (img_real-np.mean(img_real))/np.std(img_real)

synapse_color = norm_img_real[x_synapse,y_synapse]
print np.mean(synapse_color)
print np.std(synapse_color)

plt.figure()
plt.imshow(img,cmap=plt.cm.gray)
#for n in xrange(img.shape[0]):
#    for m in xrange(img.shape[1]):
#        if img[n,m] == 1:
#            if norm_img_real[n,m]>0.:
#                img[n,m] = 0
plt.figure()
plt.imshow(norm_img_real,cmap=plt.cm.gray)
plt.colorbar()

sobelx = cv2.Sobel(norm_img_real,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(norm_img_real,cv2.CV_64F,0,1,ksize=5)
sobel  = np.sqrt(sobelx**2+sobely**2)
fourier = np.fft.fft2(sobel)
#plt.figure()
#n, bins, patches = plt.hist(np.real(fourier).flatten(1), 10000, normed=1, histtype='stepfilled')
#n, bins, patches = plt.hist(np.imag(fourier).flatten(1), 10000, normed=1, histtype='stepfilled')
#
#plt.show()
#exit()
kernel = np.ones((12,12),np.uint8)
plt.figure()
plt.imshow(sobel,cmap=plt.cm.gray)
#n, bins, patches = plt.hist(synapse_color, 50, normed=1, histtype='stepfilled')
plt.show()

exit()

plt.figure(1)
plt.imshow(img_real,cmap=plt.cm.gray)
plt.figure(2)
plt.imshow(img,cmap = plt.cm.gray)
plt.show()
