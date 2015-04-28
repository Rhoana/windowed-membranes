import numpy as np
import matplotlib.pyplot as plt

output1 = np.load('synapse1/output.npy')
output2 = np.load('synapse2/output.npy')
#output3 = np.load('latest3/output.npy')

y1 = np.load('synapse1/y.npy')
y2 = np.load('synapse2/y.npy')
#y3 = np.load('latest3/y.npy')

output = (output1+output2)/float(2)
y = (y1+y2)/float(2)

print np.mean(np.abs(output1-y1))
print np.mean(np.abs(output2-y1))
print np.mean(np.abs(output-y))
exit()

plt.figure()
plt.imshow(output1[2],cmap=plt.cm.gray)
plt.figure()
plt.imshow(y1[2],cmap =plt.cm.gray)
plt.show()
