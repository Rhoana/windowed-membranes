import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import sys
import util.post_process as post
from sklearn.metrics import f1_score

def plot_samples(n=0):
    x = np.load('results/x.npy')
    y = np.load('results/y.npy')
    output = np.load('results/output.npy')

    difference     = x.shape[1]-output.shape[1]
    out_windowsize = output.shape[1]

    fig = plt.figure(1)
    ax1 = fig.add_subplot(131)
    ax1.imshow(x[n,:,:],cmap=plt.cm.gray)
    ax1.add_patch(Rectangle((difference/2, difference/2), out_windowsize, out_windowsize, fill=None, alpha=1))

    ax2 = fig.add_subplot(133)
    ax2.imshow(output[n,:,:],cmap=plt.cm.gray)

    ax3 = fig.add_subplot(132)
    ax3.imshow(y[n,:,:],cmap=plt.cm.gray)
    print np.unique(y)

    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)


    ax1.set_title('Input Image')
    ax3.set_title('Labeled Edges')
    ax2.set_title('Predicted Edges')

def plot(n=0):

    n = int(n)
    
    post.post_process()

    y      = np.load('results/y_whole.npy')
    output = np.load('results/output_whole.npy')

    print 'Max/min y-value(bug-check): ',y.max(),y.min()
    print 'Max/min pred-value(bug-check): ',output.max(),output.min()

    print f1_score(np.round(output.flatten(1)),y.flatten(1))

    plt.figure(1)
    plt.imshow(y[n],cmap=plt.cm.gray)
    plt.figure(2)
    plt.imshow(output[n],cmap=plt.cm.gray)



    
    plt.show()
    
if __name__ == "__main__":
    
    if len(sys.argv)>1:
        n = sys.argv[1]
    else:
        n = 0
    
    plot(n=n)
    #plot_samples(n=n)
    plt.show()
