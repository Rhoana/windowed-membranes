import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from scipy import signal

adress = 'train-labels/train_labels_cleaned_flipped_0000.tif'
img = misc.imread(adress)

#plt.imshow(img)


def find_edges(img)
    scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],[-10+0j, 0+ 0j, +10 +0j],[ -3+3j, 0+10j,  +3 +3j]])
    grad = signal.convolve2d(img, scharr, boundary='symm', mode='same')

    grad = np.absolute(grad).astype(np.float32)

    for n in xrange(grad.shape[0]):
        for m in xrange(grad.shape[1]):
            if grad[n,m] >0.:
                grad[n,m] = 1.

    return img

plt.imshow(grad.astype(np.int32),cmap= plt.cm.gray)
plt.show()
