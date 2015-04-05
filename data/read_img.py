#File for generating samples from input data
##

import numpy as np
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

def sample(x,y,imarray,thick_edged,input_image,find_number,in_window_shape,out_window_shape,img_size=(1024,1024),n_samples=10,random=False):
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

def define_arrays(directory_input,directory_labels,samples_per_image,in_window_shape,out_window_shape,n_test_files=1):
    
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
        
        img_real = cv2.imread(adress_real_img,cv2.IMREAD_UNCHANGED)
        img = cv2.imread(adress,cv2.IMREAD_UNCHANGED)
        unique = np.unique(img)
        img_copy = np.uint8(img)
        thin_edged = find_edges(img_copy)
        thick_edged = thick_edge(thin_edged)
        
        x_temp,y_temp = np.zeros((0,in_window_shape[0]*in_window_shape[1])),np.zeros((0,out_window_shape[0]*out_window_shape[1])) 
        x_temp,y_temp = sample(x_temp,y_temp,thin_edged,thick_edged,img_real,1,in_window_shape=in_window_shape,out_window_shape=out_window_shape,n_samples=samples_per_image/2)
        
        x = np.vstack((x,x_temp))
        y = np.vstack((y,y_temp))

        x_temp,y_temp = np.zeros((0,in_window_shape[0]*in_window_shape[1])),np.zeros((0,out_window_shape[0]*out_window_shape[1])) 
        x_temp,y_temp = sample(x_temp,y_temp,thin_edged,thick_edged,img_real,0,in_window_shape=in_window_shape,out_window_shape=out_window_shape,n_samples=samples_per_image/2)

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
        
        img_real = cv2.imread(adress_real_img,cv2.IMREAD_UNCHANGED)
        img = cv2.imread(adress,cv2.IMREAD_UNCHANGED)
        img_copy = np.uint8(img)
        thin_edged = find_edges(img_copy)
        thick_edged = thick_edge(thin_edged)

        img_samples,labels,table_temp = generate_test_set(thick_edged,img_real,in_window_shape,out_window_shape,n)

        x     = np.vstack((x,img_samples))
        y     = np.vstack((y,labels))
        table = np.vstack((table,table_temp))

    np.save('data/x_test.npy',x)
    np.save('data/y_test.npy',y)
    np.save('data/table.npy',table)
    
    print 'Done ... '
    

def generate_test_set(thick_edged, img_real, in_window_shape,out_window_shape, img_number,img_shape = (1024,1024)):
    offset = in_window_shape[0]/2
    diff  = (in_window_shape[0]-out_window_shape[0])

    thick_edged = thick_edged[(diff/2):-(diff/2),(diff/2):-(diff/2)]
    
    number = thick_edged.shape[0]/out_window_shape[0]

    img_samples = np.zeros((number**2,in_window_shape[0]**2))
    labels = np.zeros((number**2,out_window_shape[0]**2))
    table = np.zeros((number**2,3),dtype=np.int32)

    start_point = diff

    table_number = 0
    for n in xrange(number-1):
        for m in xrange(number-1):
            img_samples[table_number,:] = img_real[(out_window_shape[0]*n):(out_window_shape[0]*(n+1)+diff),(out_window_shape[0]*m):(out_window_shape[0]*(m+1)+diff)].reshape(1,in_window_shape[0]**2)
            
            labels[table_number,:] = thick_edged[(out_window_shape[0]*n):(out_window_shape[0]*(n+1)),(out_window_shape[0]*m):(out_window_shape[0]*(m+1))].reshape(1,out_window_shape[0]**2)
            table[table_number,0] = img_number
            table[table_number,1] = n
            table[table_number,2] = m
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

def generate_training_set(in_window_shape,out_window_shape,samples_per_image = 200):

    # Define directory input and arrays
    directory_input = 'data/train-input'
    directory_labels = 'data/train-labels'
    define_arrays(directory_input,directory_labels,samples_per_image,in_window_shape,out_window_shape)
    
if __name__ == '__main__':
    generate_training_set((64,64),(12,12))
   
    
    
