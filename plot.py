import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import sys
import util.post_process as post

def plot_samples(n=0):
    x = np.load('results/x_train_examples.npy')
    y = np.load('results/y_train_examples.npy')

    window_size_x = int(np.sqrt(x.shape[1]))
    window_size_y = int(np.sqrt(y.shape[1]))

    x = x.reshape(x.shape[0],window_size_x,window_size_x)
    y = y.reshape(y.shape[0],window_size_y,window_size_y)

    difference     = x.shape[1]-y.shape[1]
    out_windowsize = y.shape[1]

    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax1.imshow(x[n,:,:],cmap=plt.cm.gray)
    ax1.add_patch(Rectangle((difference/2, difference/2), out_windowsize, out_windowsize, fill=None, alpha=1))

    ax2 = fig.add_subplot(122)
    ax2.imshow(y[n,:,:],cmap=plt.cm.gray)

    print y[n,:,:]

    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    ax1.set_title('Input Image')
    ax2.set_title('Labeled')

def plot(n=0):
    n = int(n)
    
    y      = np.load('results/y.npy')
    output = np.load('results/output.npy')

    print 'Max/min y-value(bug-check): ',y.max(),y.min()
    print 'Max/min pred-value(bug-check): ',output.max(),output.min()

    plt.figure(1)
    plt.imshow(y[n],cmap=plt.cm.gray)
    plt.colorbar()
    plt.figure(2)
    plt.imshow(output[n],cmap=plt.cm.gray)
    plt.colorbar()
    plt.show()

def results():
    
    results = np.load('results/results.npy')

    fig = plt.figure(1)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.set_title('Cost and validation error')
    ax1.set_ylabel('Training cost')
    ax2.set_ylabel('Validation error')
    ax2.set_xlabel('Epochs')

    ax1.plot(results[0],color = 'blue')
    ax2.plot(results[1],color = 'red')
    plt.figure(2)
    plt.plot(results[2])
    plt.show()

if __name__ == "__main__":
    

    if "--results" in sys.argv:
        results()

    elif "--samples" in sys.argv:
        if len(sys.argv)>2:
            n = sys.argv[-1]
        else:
            n = 0
        plot_samples(n=n)
    else:
        if len(sys.argv)>1:
            n = sys.argv[-1]
        else:
            n = 0
        plot(n=n)
    plt.show()
