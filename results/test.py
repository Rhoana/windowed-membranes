import numpy as np
import matplotlib.pyplot as plt

output1 = np.load('latest1/output.npy')
output2 = np.load('latest2/output.npy')
output3 = np.load('latest3/output.npy')

y1 = np.load('latest1/y.npy')
y2 = np.load('latest2/y.npy')
y3 = np.load('latest3/y.npy')

output = (output1+output2+output3)/float(3)
y = (y1+y2+y3)/float(3)

plt.figure()
plt.imshow(output1[0],cmap=plt.cm.gray)
plt.figure()
plt.imshow(y1[0],cmap =plt.cm.gray)
plt.show()
