import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image

adress = 'AC3/AC3_SynTruthVolume.tif'
img = Image.open(adress)

img_stack = np.zeros((0,1024*1024))

flag = True
i = 0
while flag == True:
    try:
        img.seek(i)
        img_stack = np.vstack((img_stack,np.asarray(img).flatten(1)))
        i += 1
    except EOFError:
        flag = False

print img_stack.shape

def find_edges(img):
    scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],[-10+0j, 0+ 0j, +10 +0j],[ -3+3j, 0+10j,  +3 +3j]])
    grad = signal.convolve2d(img, scharr, boundary='symm', mode='same')

    grad = np.absolute(grad).astype(np.float32)

    for n in xrange(grad.shape[0]):
        for m in xrange(grad.shape[1]):
            if grad[n,m] >0.:
                grad[n,m] = 1.

    return img

print img_stack.shape
#plt.imshow(grad.astype(np.int32),cmap= plt.cm.gray)
plt.imshow(img,cmap=plt.cm.gray)
plt.show()
