import numpy as np
import matplotlib.pyplot as plt

x = np.load('x_train.npy')
y = np.load('y_train.npy')

n = 12
plt.imshow(x[n].reshape(64,64),cmap= plt.cm.gray)
plt.figure()
plt.imshow(y[n].reshape(48,48),cmap= plt.cm.gray)
plt.show()
