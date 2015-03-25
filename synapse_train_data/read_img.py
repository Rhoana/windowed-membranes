## File for generating samples from input data
##
##

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os 
    
def find_edges(img):
    edged = cv2.Canny(img,1,1) 
    edged = convert_binary(edged)
    return edged

def convert_binary(imarray):
    for n in xrange(imarray.shape[0]):
        for m in xrange(imarray.shape[1]):
            if imarray[n,m] >0:
                imarray[n,m] = 1
    return imarray

def sample(x,y,imarray,thick_edged,input_image,find_number,img_size=(1024,1024),windowsize=(48,48),n_samples=10):
    offset = windowsize[0]/2-1
    temp = imarray[offset:img_size[0]-offset-1,offset:img_size[0]-offset-1]
    
    ix = np.in1d(temp.ravel(), find_number).reshape(temp.shape)
    temp_edge_x,temp_edge_y = np.where(ix)
    rand = np.random.permutation(range(temp_edge_x.size))
    
    rand = rand[:n_samples]

    edge_x = np.zeros(rand.size)
    edge_y = np.zeros(rand.size)
    for n in xrange(rand.size):
        edge_x[n] = temp_edge_x[rand[n]]
        edge_y[n] = temp_edge_y[rand[n]]

    for n in xrange(edge_x.size):
        start_point_x = edge_x[n]
        end_point_x   = start_point_x+windowsize[0]
        start_point_y = edge_y[n]
        end_point_y   = start_point_y+windowsize[1]

        edge_sample = thick_edged[start_point_x:end_point_x,start_point_y:end_point_y]
        image_sample = input_image[start_point_x:end_point_x,start_point_y:end_point_y]

        edge_sample = edge_sample.reshape(windowsize[0]*windowsize[1],)
        image_sample = image_sample.reshape(windowsize[0]*windowsize[1],)

        x = np.vstack((x,image_sample))
        y = np.vstack((y,edge_sample))
    return x,y

def define_arrays(directory_input,directory_labels,samples_per_image,windowsize = (48,48)):
    
    print('Defining input ...')

    files_input = glob.glob(directory_input+"/*.tif")
    files_labels = glob.glob(directory_labels+"/*.tif")
    x = np.zeros((0,windowsize[0]*windowsize[1]))
    y = np.zeros((0,windowsize[0]*windowsize[1]))
    
    n = 0
    for n in range(len(files_input)):
        print 'Processing file '+str(n+1) + '... '
        adress_real_img = files_input[n]
        adress = files_labels[n]
        
        img_real = cv2.imread(adress_real_img,cv2.IMREAD_UNCHANGED)
        img = cv2.imread(adress,cv2.IMREAD_UNCHANGED)
        img_copy = np.uint8(img)
        thin_edged = find_edges(img_copy)
        thick_edged = thick_edge(thin_edged)
        
        x_temp,y_temp = np.zeros((0,windowsize[0]*windowsize[1])),np.zeros((0,windowsize[0]*windowsize[1])) 
        x_temp,y_temp = sample(x_temp,y_temp,thin_edged,thick_edged,img_real,1,n_samples=samples_per_image/2)
        x_temp,y_temp = sample(x_temp,y_temp,thin_edged,thick_edged,img_real,0,n_samples=samples_per_image/2)
        
        x = np.vstack((x,x_temp))
        y = np.vstack((y,y_temp))

    print 'Done ... '
    
    return x,y

def thick_edge(imarray):
    thickarray = np.zeros(np.shape(imarray))

    for n in xrange(1,imarray.shape[0]-1):
        for m in xrange(1,imarray.shape[1]-1):
            if imarray[n,m] == 1:
                thickarray[n,m] = 1
                thickarray[n-1,m] = 1
                thickarray[n-1,m] = 1
                thickarray[n,m+1] = 1
                thickarray[n,m-1] = 1
                thickarray[n-1,m-1] = 1
                thickarray[n-1,m+1] = 1
                thickarray[n+1,m+1] = 1
                thickarray[n+1,m-1] = 1
    return thickarray

def show_example(adress_real,adress_label):
    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    img = cv2.imread(adress_label,cv2.IMREAD_UNCHANGED)
    ax1.imshow(img,cmap=plt.cm.gray)
    img = np.uint8(img)
    img = find_edges(img)
    ax2 = fig.add_subplot(122)
    img = thick_edge(img)
    ax2.imshow(img,cmap=plt.cm.gray)

    ax1.set_title('Labeled Input')
    ax2.set_title('Find Edges')

    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    fig.savefig('pres2.png')
    plt.show()
    exit()

def generate_training_set(samples_per_image = 100):
    
    # Define directory input and arrays
    directory_input = 'synapse_train_data/train-input'
    directory_labels = 'synapse_train_data/train-labels'
    x,y = define_arrays(directory_input,directory_labels,samples_per_image)
    
    print 'Size dataset: ',x.shape,y.shape

    # Save arrays
    np.save('synapse_train_data/x.npy',x)
    np.save('synapse_train_data/y.npy',y)

if __name__ == '__main__':
    generate_training_set()
   
    
    
