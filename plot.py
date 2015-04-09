import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import sys

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

    img_shape = (1024,1024)                                                         
    in_window_shape = (64,64)                                                       
    out_window_shape = (12,12)                                                      
    diff = in_window_shape[0]-out_window_shape[0]    

    output = np.load('results/output.npy')
    y     = np.load('results/y.npy')                                              
    table = np.load('data/table.npy') 

    #output = np.round(output/np.max(output))

    y = y.reshape(y.shape[0],out_window_shape[0],out_window_shape[1])               
    img = np.zeros((img_shape[0]-diff,img_shape[0]-diff))
    for i in xrange(table.shape[0]):    
        if table[i,0] == n:       
            img[(table[i,1]*out_window_shape[0]):((table[i,1]+1)*out_window_shape[0]),(table[i,2]*out_window_shape[0]):((table[i,2]+1)*           out_window_shape[0])]= y[i]                                                    

    plt.figure(1)
    plt.imshow(img,cmap=plt.cm.gray)

    output = output.reshape(output.shape[0],out_window_shape[0],out_window_shape[1])               
    img = np.zeros((img_shape[0]-diff,img_shape[0]-diff))
    for i in xrange(table.shape[0]):    
        if table[i,0] == n:       
            img[(table[i,1]*out_window_shape[0]):((table[i,1]+1)*out_window_shape[0]),(table[i,2]*out_window_shape[0]):((table[i,2]+1)*           out_window_shape[0])]= output[i]                                                    

    plt.figure(2)
    plt.imshow(img,cmap=plt.cm.gray)

    plt.show()

    
if __name__ == "__main__":
    
    if len(sys.argv)>1:
        n = sys.argv[1]
    else:
        n = 0
    
    plot(n=n)
    #plot_samples(n=n)
    plt.show()
