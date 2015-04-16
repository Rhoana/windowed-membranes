#File for generating samples from input data
##

import numpy as np
import glob
import os
from scipy import signal
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc


def edge_filter(img):
    scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],[-10+0j, 0+ 0j, +10 +0j],[ -3+3j, 0+10j,  +3 +3j]])
    grad = signal.convolve2d(img, scharr, boundary='symm', mode='same')         
                                        
    grad = np.absolute(grad).astype(np.float32)
    grad = grad/np.max(grad)
                                            
    return grad    

def find_edges(img):

    threshold = 399
    # Remove synapsis
    for n in xrange(img.shape[0]):
        for m in xrange(img.shape[1]):
            if img[n,m] > threshold:
                img[n,m] = 0

    img = edge_filter(img)

    edged = convert_binary(img)
    return edged

def find_synapse(img,edges = False):
    threshold = 399

    # Find synapse
    for n in xrange(img.shape[0]):
        for m in xrange(img.shape[1]):
            if img[n,m] > threshold:
                img[n,m] = 1
            else:
                img[n,m] = 0

    if edges == True:
        print 'Error - implement Gaussian blur for synapse'
        exit()

    else:
        blur_img = img
    
    return img,blur_img

def convert_binary(imarray):
    for n in xrange(imarray.shape[0]):
        for m in xrange(imarray.shape[1]):
            if imarray[n,m] >0:
                imarray[n,m] = 1
    return imarray

def sample(x,y,imarray,thick_edged,input_image,find_number,in_window_shape,out_window_shape,img_size=(1024,1024),n_samples=100,random=False):
    offset = (in_window_shape[0]/2)
    temp = imarray[offset:-(offset),offset:-(offset)]

    if random ==False:
        ix = np.in1d(temp.ravel(), find_number).reshape(temp.shape)
        temp_edge_x,temp_edge_y = np.where(ix)
        rand = np.random.permutation(range(temp_edge_x.size))
        rand = rand[:n_samples]

        edge_x = np.zeros(rand.size)
        edge_y = np.zeros(rand.size)
        for n in xrange(rand.size):
            edge_x[n] = temp_edge_x[rand[n]]
            edge_y[n] = temp_edge_y[rand[n]]

    elif random ==True:
        rand = np.random.permutation(range(temp.shape[0]))
        edge_x = rand[:n_samples]
        edge_y = rand[:n_samples]

        rand = rand[:n_samples]

    for n in xrange(rand.size):
        in_start_point_x = edge_x[n]
        in_end_point_x = in_start_point_x + in_window_shape[0]
        in_start_point_y = edge_y[n]
        in_end_point_y = in_start_point_y + in_window_shape[1]
        out_start_point_x = edge_x[n] + (in_window_shape[0]-out_window_shape[0])/2
        out_end_point_x = out_start_point_x + out_window_shape[0]
        out_start_point_y = edge_y[n] + (in_window_shape[1]-out_window_shape[1])/2
        out_end_point_y = out_start_point_y + out_window_shape[1]

        edge_sample = thick_edged[out_start_point_x:out_end_point_x,out_start_point_y:out_end_point_y]
        image_sample = input_image[in_start_point_x:in_end_point_x,in_start_point_y:in_end_point_y]

        edge_sample = edge_sample.reshape(out_window_shape[0]*out_window_shape[1],)
        image_sample = image_sample.reshape(in_window_shape[0]*in_window_shape[1],)

        x = np.vstack((x,image_sample))
        y = np.vstack((y,edge_sample))
    return x,y

def define_arrays(directory_input,directory_labels,samples_per_image,in_window_shape,out_window_shape,membrane,synapse,stride,n_test_files=5, gaussian_blur = True, on_ratio = 0.5, sigma = 3):
    
    print('Defining input ...')

    files_input = glob.glob(directory_input+"/*.tif")
    files_labels = glob.glob(directory_labels+"/*.tif")
    total_files = len(files_input)
    train_files_input  = files_input[:(total_files-n_test_files)]
    train_files_labels = files_labels[:(total_files-n_test_files)]
    test_files_input  = files_input[(total_files-n_test_files):]
    test_files_labels = files_labels[(total_files-n_test_files):]

    x = np.zeros((0,in_window_shape[0]*in_window_shape[1]))
    y = np.zeros((0,out_window_shape[0]*out_window_shape[1]))
    
    n = 0
    for n in range(len(train_files_input)):
	print 'Processing file '+str(n+1) + '... '
        adress_real_img = train_files_input[n]
        adress = train_files_labels[n]
        
        img_real = plt.imread(adress_real_img)
        img = plt.imread(adress)

        if membrane == True:
            thin_edged = find_edges(img)
            thin_edged = thin_edged/thin_edged.max()

            if gaussian_blur != True:
                thick_edged = thick_edge(thin_edged)

            else:
                thick_edged = scipy.ndimage.gaussian_filter(thin_edged, sigma=sigma)
                thick_edged = thick_edged/thick_edged.max()
            
            x_temp,y_temp = np.zeros((0,in_window_shape[0]*in_window_shape[1])),np.zeros((0,out_window_shape[0]*out_window_shape[1])) 
            x_temp,y_temp = sample(x_temp,y_temp,thin_edged,thick_edged,img_real,1,in_window_shape=in_window_shape,out_window_shape=out_window_shape,n_samples=samples_per_image/2)
            
            x = np.vstack((x,x_temp))
            y = np.vstack((y,y_temp))

            x_temp,y_temp = np.zeros((0,in_window_shape[0]*in_window_shape[1])),np.zeros((0,out_window_shape[0]*out_window_shape[1])) 
            x_temp,y_temp = sample(x_temp,y_temp,thin_edged,thick_edged,img_real,0,in_window_shape=in_window_shape,out_window_shape=out_window_shape,n_samples=samples_per_image/2)

            x = np.vstack((x,x_temp))
            y = np.vstack((y,y_temp))

        elif synapse == True:
            on_synapse  = samples_per_image *on_ratio
            off_synapse = samples_per_image *(1-on_ratio)

            synapsis,blur_synapsis = find_synapse(img)

            x_temp,y_temp = np.zeros((0,in_window_shape[0]*in_window_shape[1])),np.zeros((0,out_window_shape[0]*out_window_shape[1])) 
            x_temp,y_temp = sample(x_temp,y_temp,synapsis,blur_synapsis,img_real,1,in_window_shape=in_window_shape,out_window_shape=out_window_shape,n_samples=on_synapse)
            
            x = np.vstack((x,x_temp))
            y = np.vstack((y,y_temp))

            x_temp,y_temp = np.zeros((0,in_window_shape[0]*in_window_shape[1])),np.zeros((0,out_window_shape[0]*out_window_shape[1])) 
            x_temp,y_temp = sample(x_temp,y_temp,synapsis,blur_synapsis,img_real,0,in_window_shape=in_window_shape,out_window_shape=out_window_shape,n_samples=off_synapse)

            x = np.vstack((x,x_temp))
            y = np.vstack((y,y_temp))

    np.save('data/x_train.npy',x)
    np.save('data/y_train.npy',y)

    x = np.zeros((0,in_window_shape[0]*in_window_shape[1]))
    y = np.zeros((0,out_window_shape[0]*out_window_shape[1]))
    table = np.zeros((0,3))

    m = n+1

    for n in range(len(test_files_input)):
        print 'Processing file '+str(n+1+m) + '... '
        adress_real_img = test_files_input[n]
        adress = test_files_labels[n]
        
        img_real = plt.imread(adress_real_img)
        img = plt.imread(adress)

        if membrane == True:
            thin_edged = find_edges(img)
            thin_edged = thin_edged/thin_edged.max()

            if gaussian_blur != True:
                thick_edged = thick_edge(thin_edged)

            else:
                thick_edged = scipy.ndimage.gaussian_filter(thin_edged, sigma=sigma)
                thick_edged = thick_edged/thick_edged.max()  
   
            img_samples,labels,table_temp = generate_test_set(thick_edged,img_real,in_window_shape,out_window_shape,n,stride)

            x     = np.vstack((x,img_samples))
            y     = np.vstack((y,labels))
            table = np.vstack((table,table_temp))

        elif synapse == True:
            synapsis,blur_synapsis = find_synapse(img)

            img_samples,labels,table_temp = generate_test_set(blur_synapsis,img_real,in_window_shape,out_window_shape,n,stride)

            x     = np.vstack((x,img_samples))
            y     = np.vstack((y,labels))
            table = np.vstack((table,table_temp))

    np.save('data/x_test.npy',x)
    np.save('data/y_test.npy',y)
    np.save('data/table.npy',table)
    
    print 'Done ... '

def generate_test_set(thick_edged, img_real, in_window_shape,out_window_shape, img_number,stride,img_shape = (1024,1024)):
    
    offset = in_window_shape[0]/2
    diff  = (in_window_shape[0]-out_window_shape[0])

    thick_edged = thick_edged[(diff/2):-(diff/2),(diff/2):-(diff/2)]
    
    number = thick_edged.shape[0]/stride - (out_window_shape[0]/stride - 1)

    img_samples = np.zeros((number**2,in_window_shape[0]**2))
    labels = np.zeros((number**2,out_window_shape[0]**2))
    table = np.zeros((number**2,3),dtype=np.int32)

    table_number = 0
    for n in xrange(number):
        for m in xrange(number):

            img_start_y = stride*n
            img_end_y   = stride*n + in_window_shape[0]
            img_start_x = stride*m
            img_end_x   = stride*m + in_window_shape[0]

            label_start_y = stride*n
            label_end_y   = stride*n + out_window_shape[0]
            label_start_x = stride*m
            label_end_x   = stride*m + out_window_shape[0]

            img_samples[table_number,:] = img_real[img_start_y:img_end_y,img_start_x:img_end_x].reshape(1,in_window_shape[0]**2)
            labels[table_number,:]      = thick_edged[label_start_y:label_end_y, label_start_x:label_end_x].reshape(1,out_window_shape[0]**2)

            table[table_number,0] = img_number
            table[table_number,1] = label_start_y
            table[table_number,2] = label_start_x
            table_number += 1

    return img_samples,labels,table

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

def generate_data(in_window_shape,out_window_shape,samples_per_image = 200,membrane = True, synapse = False,stride = 12):

    # Define directory input and arrays
    directory_input = 'data/train-input'
    directory_labels = 'data/train-labels'
    define_arrays(directory_input,directory_labels,samples_per_image,in_window_shape,out_window_shape,membrane,synapse,stride)
    
if __name__ == '__main__':
    generate_training_set((64,64),(12,12))
   
    
    
