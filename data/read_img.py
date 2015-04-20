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
from PIL import Image 

class Read(object):

    def __init__(self,in_window_shape, out_window_shape, stride, img_size, classifier, n_train_files, n_test_files, samples_per_image, on_ratio, directory_input, directory_labels, membrane_edges):
        self.in_window_shape   = in_window_shape
        self.out_window_shape  = out_window_shape 
        self.stride            = stride
        self.img_size          = img_size
        self.n_train_files     = n_train_files
        self.n_test_files      = n_test_files
        self.samples_per_image = samples_per_image
        self.on_ratio          = on_ratio
        self.classifier        = classifier 
        self.directory_input   = directory_input
        self.directory_labels  = directory_labels

        self.membrane_edges     = membrane_edges
        self.sigma              = 3 

    def edge_filter(self,img):
        scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],[-10+0j, 0+ 0j, +10 +0j],[ -3+3j, 0+10j,  +3 +3j]])
        grad = signal.convolve2d(img, scharr, boundary='symm', mode='same')         
                                            
        grad = np.absolute(grad).astype(np.float32)
        grad = grad/np.max(grad)
                                                
        return grad    

    def find_edges(self,img):

        threshold = 399
        # Remove synapsis
        for n in xrange(img.shape[0]):
            for m in xrange(img.shape[1]):
                if img[n,m] > threshold:
                    img[n,m] = 0

        img = self.edge_filter(img)

        edged = self.convert_binary(img)
        return edged

    def find_synapse(self,img,File,edges = False):
        
        if File != 'data/AC3-labels/AC3_SynTruthVolume.tif':
            threshold = 399
        if File == 'data/AC3-labels/AC3_SynTruthVolume.tif':
            threshold = 0


        if File == 'AC3_SynTruthVolume.tif':
            import matplotlib.pyplot as plt
            plt.imshow(img,cmap=plt.cm.gray)
            plt.show()

        # Find synapse
        for n in xrange(img.shape[0]):
            for m in xrange(img.shape[1]):
                if img[n,m] > threshold:
                    img[n,m] = 1
                else:
                    img[n,m] = 0

        return img

    def convert_binary(self,imarray):
        for n in xrange(imarray.shape[0]):
            for m in xrange(imarray.shape[1]):
                if imarray[n,m] >0:
                    imarray[n,m] = 1
        return imarray

    def sample(self,x,y,imarray,thick_edged,input_image,n_samples,sample_stride = 6,find_number = 0,on_synapse = False,diff_samples = 0,on_synapse_threshold = 0.1):

        n_samples -= diff_samples

        if self.classifier != 'synapse_reg':
            offset = (self.in_window_shape[0]/2)
            temp = imarray[offset:-(offset),offset:-(offset)]
            temp = temp[::sample_stride,::sample_stride]

            ix = np.in1d(temp.ravel(), find_number).reshape(temp.shape)
            temp_edge_x,temp_edge_y = np.where(ix)
            temp_edge_x.flags.writeable = True
            temp_edge_y.flags.writeable = True
            temp_edge_x *= sample_stride
            temp_edge_y *= sample_stride

            if temp_edge_x.size < n_samples:
                print 'Warning: Not enough samples...'
                diff_samples = n_samples-temp_edge_x.size
            rand = np.random.permutation(range(temp_edge_x.size))
            rand = rand[:n_samples]

            edge_x = np.zeros(rand.size)
            edge_y = np.zeros(rand.size)
            for n in xrange(rand.size):
                edge_x[n] = temp_edge_x[rand[n]]
                edge_y[n] = temp_edge_y[rand[n]]

            for n in xrange(rand.size):
                in_start_point_x = edge_x[n]
                in_end_point_x = in_start_point_x + self.in_window_shape[0]
                in_start_point_y = edge_y[n]
                in_end_point_y = in_start_point_y + self.in_window_shape[1]
                out_start_point_x = edge_x[n] + (self.in_window_shape[0]-self.out_window_shape[0])/2
                out_end_point_x = out_start_point_x + self.out_window_shape[0]
                out_start_point_y = edge_y[n] + (self.in_window_shape[1]-self.out_window_shape[1])/2
                out_end_point_y = out_start_point_y + self.out_window_shape[1]

                edge_sample = thick_edged[out_start_point_x:out_end_point_x,out_start_point_y:out_end_point_y]
                image_sample = input_image[in_start_point_x:in_end_point_x,in_start_point_y:in_end_point_y]

                edge_sample = edge_sample.reshape(self.out_window_shape[0]*self.out_window_shape[1],)
                image_sample = image_sample.reshape(self.in_window_shape[0]*self.in_window_shape[1],)

                x = np.vstack((x,image_sample))
                y = np.vstack((y,edge_sample))

        else:
            offset = (self.in_window_shape[0]-self.out_window_shape[0])/2
            temp = imarray[offset:-(offset),offset:-(offset)]
            temp_copy = temp.copy()
            scharr = np.ones(self.out_window_shape) 
            temp   = signal.convolve2d(temp, scharr, mode='valid')
            temp /= float(self.out_window_shape[0]**2)

            if on_synapse == True:
                temp_x_samples,temp_y_samples = np.where(temp > on_synapse_threshold)
            else:
                temp_x_samples,temp_y_samples = np.where(temp < on_synapse_threshold)

            if temp_x_samples.size < n_samples:
                print 'Warning: Not enough samples...',temp_x_samples.size
                diff_samples = n_samples-temp_x_samples.size

            temp_x_samples.flags.writeable = True
            temp_y_samples.flags.writeable = True
            rand = np.random.permutation(range(temp_x_samples.size))
            rand = rand[:n_samples]

            x_samples = np.zeros(rand.size)
            y_samples = np.zeros(rand.size)
            for n in xrange(rand.size):
                x_samples[n] = temp_x_samples[rand[n]]
                y_samples[n] = temp_y_samples[rand[n]]


            for n in xrange(rand.size):
                in_start_point_x = x_samples[n]
                in_end_point_x   = in_start_point_x + self.in_window_shape[0]
                in_start_point_y = y_samples[n]
                in_end_point_y   = in_start_point_y + self.in_window_shape[1]

                out_sample = temp[x_samples[n],y_samples[n]]
                temp_copy_sample  = temp_copy[x_samples[n]:(x_samples[n]+self.out_window_shape[0]),y_samples[n]:(y_samples[n]+self.out_window_shape[0])]
                image_sample = input_image[in_start_point_x:in_end_point_x,in_start_point_y:in_end_point_y]

                out_sample = out_sample.reshape(1,)
                image_sample = image_sample.reshape(self.in_window_shape[0]*self.in_window_shape[1],)

                x = np.vstack((x,image_sample))
                y = np.append(y,out_sample)

        return x,y,diff_samples

    def define_arrays(self,directory_input,directory_labels):
        
        print('Defining input ...')

        files_input  = []
        files_labels = []

        for directory in directory_input:
            files_input = files_input + sorted(glob.glob(directory+"/*.tif"))
        for directory in directory_labels:
            files_labels = files_labels + sorted(glob.glob(directory+"/*.tif"))
        img_real_stack = np.zeros((0,self.img_size[0]**2))

        for File in files_input:
            img_temp = Image.open(File)                                                        
                                                
            flag = True                                                                     
            i = 0                                                                           
            while flag == True:                                                             
                try:                                                                        
                    img_temp.seek(i)                                                             
                    img_real_stack = np.vstack((img_real_stack,np.asarray(img_temp).flatten(1)))           
                    i += 1                                                                  
                except EOFError:                                                            
                    flag = False   

        img_stack = np.zeros((0,self.img_size[0]**2))

        for File in files_labels:
            img_temp = Image.open(File)                                                        
                                                
            flag = True                                                                     
            i = 0                                                                           
            while flag == True:                                                             
                try:                                                                        
                    img_temp.seek(i)
                    img_temp_temp = np.asarray(img_temp)
                    img_temp_temp.flags.writeable = True
                    if self.classifier == 'membrane':
                        img_temp_temp = self.find_edges(img_temp_temp)
                    elif self.classifier == 'synapse' or self.classifier == 'synapse_reg':
                        img_temp_temp = self.find_synapse(img_temp_temp,File)

                    img_stack = np.vstack((img_stack,img_temp_temp.flatten(1)))           
                    i += 1                                                                  
                except EOFError:                                                            
                    flag = False   

        total_files = img_stack.shape[0]

        rand = np.random.permutation(range(total_files))
        img_real_stack = img_real_stack[rand]
        img_stack = img_stack[rand]

        if self.n_train_files == None:
            train_img_input  = img_real_stack[:(total_files-self.n_test_files)]
            train_img_labels = img_stack[:(total_files-self.n_test_files)]
        else:
            train_img_input  = img_real_stack[:self.n_train_files]
            train_img_labels = img_stack[:self.n_train_files]

        test_img_input  = img_real_stack[(total_files-self.n_test_files):]
        test_img_labels = img_stack[(total_files-self.n_test_files):]

        on  = int(self.samples_per_image *self.on_ratio)
        off = (self.samples_per_image *(1-self.on_ratio))
       
        train_img_input  = train_img_input.reshape(train_img_input.shape[0],self.img_size[0],self.img_size[1])
        train_img_labels = train_img_labels.reshape(train_img_labels.shape[0],self.img_size[0],self.img_size[1])
        test_img_input   = test_img_input.reshape(test_img_input.shape[0],self.img_size[0],self.img_size[1])
        test_img_labels  = test_img_labels.reshape(test_img_labels.shape[0],self.img_size[0],self.img_size[1])

        if self.classifier in ['membrane','synapse']:
            x = np.zeros((0,self.in_window_shape[0]*self.in_window_shape[1]))
            y = np.zeros((0,self.out_window_shape[0]*self.out_window_shape[1]))
        elif self.classifier == 'synapse_reg':
            x = np.zeros((0,self.in_window_shape[0]*self.in_window_shape[1]))
            y = np.zeros((0,1))
        else:
            print 'Error: Invalid Classifier'
            exit()
        n = 0
        for n in range(train_img_input.shape[0]):
            print 'Processing file '+str(n+1) + '... '
            
            img_real = train_img_input[n]
            img      = train_img_labels[n]

            if self.classifier == 'membrane':
                labeled_in = img
                labeled_in = labeled_in/labeled_in.max()

                if self.membrane_edges == 'WideEdges':
                    labeled_out = self.thick_edge(labeled_in)

                elif self.membrane_edges == 'GaussianBlur':
                    labeled_out = scipy.ndimage.gaussian_filter(labeled_in, sigma=self.sigma)
                    labeled_out = labeled_out/labeled_out.max()
                else:
                    labeled_out = labeled_in
                    print "Warning: thin edge"

            elif self.classifier == 'synapse':
                labeled_in,labeled_out = img,img

            elif self.classifier == 'synapse_reg':
                labeled_in,labeled_out = img,img

            if self.classifier in ['membrane','synapse']:
                x_temp,y_temp = np.zeros((0,self.in_window_shape[0]*self.in_window_shape[1])),np.zeros((0,self.out_window_shape[0]*self.out_window_shape[1])) 
                x_temp,y_temp,diff_samples = self.sample(x_temp,y_temp,labeled_in,labeled_out,img_real,on,find_number = 1)
                
                x = np.vstack((x,x_temp))
                y = np.vstack((y,y_temp))

                x_temp,y_temp = np.zeros((0,self.in_window_shape[0]*self.in_window_shape[1])),np.zeros((0,self.out_window_shape[0]*self.out_window_shape[1])) 
                x_temp,y_temp,diff_samples = self.sample(x_temp,y_temp,labeled_in,labeled_out,img_real,off, find_number = 0,diff_samples=diff_samples)

                x = np.vstack((x,x_temp))
                y = np.vstack((y,y_temp))

            elif self.classifier == 'synapse_reg':
                x_temp,y_temp = np.zeros((0,self.in_window_shape[0]*self.in_window_shape[1])),np.zeros((0,self.out_window_shape[0]*self.out_window_shape[1])) 
                x_temp,y_temp,diff_samples = self.sample(x_temp,y_temp,labeled_in,labeled_out,img_real,on,on,on_synapse = True)
                
                x = np.vstack((x,x_temp))
                y = np.append(y,y_temp)

                x_temp,y_temp = np.zeros((0,self.in_window_shape[0]*self.in_window_shape[1])),np.zeros((0,self.out_window_shape[0]*self.out_window_shape[1])) 
                x_temp,y_temp,diff_samples = self.sample(x_temp,y_temp,labeled_in,labeled_out,img_real,off,off,diff_samples=diff_samples,on_synapse=False)

                x = np.vstack((x,x_temp))
                y = np.append(y,y_temp)
            else:
                print 'Error: Invalid Classifier'
                exit()

        np.save('data/x_train.npy',x)
        np.save('data/y_train.npy',y)

        if self.classifier in ['synaspe','membrane']:
            x = np.zeros((0,self.in_window_shape[0]*self.in_window_shape[1]))
            y = np.zeros((0,self.out_window_shape[0]*self.out_window_shape[1]))
            table = np.zeros((0,3))
        elif self.classifier == 'synapse_reg':
            x = np.zeros((0,self.in_window_shape[0]*self.in_window_shape[1]))
            y = np.zeros((0,1))
            table = np.zeros((0,3))
        else:
            print 'Error: Invalid Classifier'
            exit()

        m = n+1

        print "Finished train set"

        for n in range(test_img_input.shape[0]):
            print 'Processing file '+str(n+1+m) + '... '
            img_real = test_img_input[n]
            img      = test_img_labels[n]

            if self.classifier == 'membrane':
                labeled_in = img
                labeled_in = labeled_in/labeled_in.max()

                if self.membrane_edges == 'WideEdges':
                    labeled_out = self.thick_edge(labeled_in)

                elif self.membrane_edges == 'GaussianBlur':
                    labeled_out = scipy.ndimage.gaussian_filter(labeled_in, sigma=self.sigma)
                    labeled_out = labeled_out/labeled_out.max()
                else:
                    labeled_out = labeled_in
                    print "Warning: thin edge"

            elif self.classifier in ['synapse','synapse_reg']:
                labeled_in,labeled_out = img,img
            else:
                print 'Error: Invalid Classifier'

            img_samples,labels,table_temp = self.generate_test_set(labeled_out,img_real,n)

            x     = np.vstack((x,img_samples))
            y     = np.vstack((y,labels))
            table = np.vstack((table,table_temp))

        np.save('data/x_test.npy',x)
        np.save('data/y_test.npy',y)
        np.save('data/table.npy',table)
        
        print 'Done ... '

    def generate_test_set(self,thick_edged, img_real, img_number):
        
        if self.classifier in ['membrane','synapse']:
            offset = self.in_window_shape[0]/2
            diff  = (self.in_window_shape[0]-self.out_window_shape[0])

            thick_edged = thick_edged[(diff/2):-(diff/2),(diff/2):-(diff/2)]
            
            number = thick_edged.shape[0]/self.stride - (self.out_window_shape[0]/self.stride - 1)

            img_samples = np.zeros((number**2,self.in_window_shape[0]**2))
            labels = np.zeros((number**2,self.out_window_shape[0]**2))
            table = np.zeros((number**2,3),dtype=np.int32)

            table_number = 0
            for n in xrange(number):
                for m in xrange(number):

                    img_start_y = self.stride*n
                    img_end_y   = self.stride*n + self.in_window_shape[0]
                    img_start_x = self.stride*m
                    img_end_x   = self.stride*m + self.in_window_shape[0]

                    label_start_y = self.stride*n
                    label_end_y   = self.stride*n + self.out_window_shape[0]
                    label_start_x = self.stride*m
                    label_end_x   = self.stride*m + self.out_window_shape[0]

                    img_samples[table_number,:] = img_real[img_start_y:img_end_y,img_start_x:img_end_x].reshape(1,self.in_window_shape[0]**2)
                    labels[table_number,:]      = thick_edged[label_start_y:label_end_y, label_start_x:label_end_x].reshape(1,self.out_window_shape[0]**2)

                    table[table_number,0] = img_number
                    table[table_number,1] = label_start_y
                    table[table_number,2] = label_start_x
                    table_number += 1

        elif self.classifier == 'synapse_reg':
            diff  = (self.in_window_shape[0]-self.out_window_shape[0])

            thick_edged = thick_edged[(diff/2):-(diff/2),(diff/2):-(diff/2)]
            
            number = thick_edged.shape[0]/self.out_window_shape[0]

            img_samples = np.zeros((number**2,self.in_window_shape[0]**2))
            labels = np.zeros((number**2,1))
            table = np.zeros((number**2,3),dtype=np.int32)

            stride = self.out_window_shape[0]
            table_number = 0
            for n in xrange(number):
                for m in xrange(number):

                    img_start_y = stride*n
                    img_end_y   = stride*n + self.in_window_shape[0]
                    img_start_x = stride*m
                    img_end_x   = stride*m + self.in_window_shape[0]

                    label_start_y = stride*n
                    label_end_y   = stride*n + self.out_window_shape[0]
                    label_start_x = stride*m
                    label_end_x   = stride*m + self.out_window_shape[0]

                    img_samples[table_number,:] = img_real[img_start_y:img_end_y,img_start_x:img_end_x].reshape(1,self.in_window_shape[0]**2)
                    labels[table_number,0]      = np.mean(thick_edged[label_start_y:label_end_y, label_start_x:label_end_x])

                    table[table_number,0] = img_number
                    table[table_number,1] = n
                    table[table_number,2] = m
                    table_number += 1

        else:
            print 'Error: Invalid Classifier'
            exit()

        return img_samples,labels,table

    def thick_edge(self,imarray):
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

    def generate_data(self):

        # Define directory input and arrays
        self.define_arrays(self.directory_input,self.directory_labels)
    
if __name__ == '__main__':
    read = Read()
    read.generate_training_set((64,64),(12,12))
   
    
    
