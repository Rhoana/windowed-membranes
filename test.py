import numpy as np

img_shape = (1024,1024)
in_window_shape = (64,64)
out_window_shape = (12,12)
diff = in_window_shape[0]-out_window_shape[0]

x     = np.load('data/x_test.npy')
y     = np.load('data/y_test.npy')
table = np.load('data/table.npy')

y = y.reshape(y.shape[0],out_window_shape[0],out_window_shape[1])

img = np.zeros((img_shape[0]-diff,img_shape[0]-diff))

print table[0:10]

for i in xrange(table.shape[0]):
    if table[i,0] != 1:
        break
    img[(table[i,1]*out_window_shape[0]):((table[i,1]+1)*out_window_shape[0]),(table[i,2]*out_window_shape[0]):((table[i,2]+1)*out_window_shape[0])]= y[i]

import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()
