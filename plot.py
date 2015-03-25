import matplotlib.pyplot as plt
import numpy as np
import sys

def plot(n=0):
    x = np.load('results/x.npy')
    y = np.load('results/y.npy')
    output = np.load('results/output.npy')

    fig = plt.figure(1)
    ax1 = fig.add_subplot(132)
    ax1.imshow(x[n].reshape(48,48),cmap=plt.cm.gray)

    ax2 = fig.add_subplot(133)
    ax2.imshow(output[n],cmap=plt.cm.gray)

    ax3 = fig.add_subplot(131)
    ax3.imshow(y[n].reshape(48,48),cmap=plt.cm.gray)

    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)


    ax3.set_title('Input Image')
    ax2.set_title('Labeled Edges')
    ax1.set_title('Predicted Edges')
    
if __name__ == "__main__":
    
    if len(sys.argv)>1:
        n = sys.argv[1]
    else:
        n = 0
        
    plot(n=n)
    plt.show()
