import numpy as np
import cv2
import matplotlib.pyplot as plt

adress = 'train-labels/train_labels_cleaned_flipped_0000.tif'

img = cv2.imread(adress,cv2.IMREAD_UNCHANGED)
img = np.uint8(img)  
img = cv2.Canny(img,1,1)
plt.imshow(img,cmap = plt.cm.gray)
plt.show()
img = cv2.GaussianBlur(img,(15,15),0)
plt.imshow(img,cmap = plt.cm.gray)
plt.show()
