import numpy as np
import matplotlib.pyplot as plt
import cv2

address = "results/48x48_av/"
pred = np.load(address + "/results/output.npy")
img = (pred[0,0]*255).astype(np.uint8)

plt.figure()
plt.imshow(img,cmap=plt.cm.gray)

ret,thresh = cv2.threshold(img,127,255,0)

import pymorph
img = pymorph.close_holes(img)

plt.figure()
plt.imshow(img,cmap=plt.cm.gray)

plt.show()
