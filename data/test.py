import numpy as np
from PIL import Image, ImageFilter

adress = 'train-labels/train_labels_cleaned_flipped_0000.tif'

img = Image.open(adress).convert("L")
arr = np.array(img)

print arr.shape

import matplotlib.pyplot as plt

plt.imshow(arr)
plt.show()
