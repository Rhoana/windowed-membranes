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
from skimage import exposure
from PIL import Image 

class Read(object):

    def __init__(self,in_window_shape, out_window_shape, stride, img_size, classifier, n_train_files, n_test_files, samples_per_image, on_ratio, directory_input, directory_labels, membrane_edges,layers_3D, adaptive_histogram_equalization):
        self.in_window_shape                 = in_window_shape
        self.out_window_shape                = out_window_shape 
        self.stride                          = stride
        self.img_size                        = img_size
        self.n_train_files                   = n_train_files
        self.n_test_files                    = n_test_files
        self.samples_per_image               = samples_per_image
        self.on_ratio                        = on_ratio
        self.classifier                      = classifier 
        self.directory_input                 = directory_input
        self.directory_labels                = directory_labels
        self.membrane_edges                  = membrane_edges
        self.sigma                           = 3 
        self.layers_3D                       = layers_3D
        self.adaptive_histogram_equalization = adaptive_histogram_equalization

    def edge_filter(self,img):
        scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],[-10+0j, 0+ 0j, +10 +0j],[ -3+3j, 0+10j,  +3 +3j]])
        grad = signal.convolve2d(img, scharr, boundary='symm', mode='same')                                                    
        grad = np.absolute(grad).astype(np.float32)
        grad = grad/np.max(grad)                             
        return grad    

    def find_edges(self,img):

        threshold = 399
        # Remove synapses
        img[np.where(img>threshold)] = 0
        img = self.edge_filter(img)

        edged = self.convert_binary(img)
        return edged

    def find_synapse(self,img,File,edges = False):
        
        if File != 'data/AC3-labels/AC3_SynTruthVolume.tif':
            threshold = 399
        if File == 'data/AC3-labels/AC3_SynTruthVolume.tif':
            threshold = 0

        if File == 'data/AC3_SynTruthVolume.tif':
            import matplotlib.pyplot as plt
            plt.imshow(img,cmap=plt.cm.gray)
            plt.show()

        # Find synapse
        img[np.where(img<threshold)] = 0
        img[np.where(img>threshold)] = 1

        return img

    def convert_binary(self,imarray):
        imarray[np.where(imarray>0)] = 1
        return imarray

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

    def sample_membrane_synapse(self,nn,imarray,thick_edged,input_image,n_samples,image_group_train,sample_stride = 4,on_membrane_synapse = False,diff_samples = 0,on_synapse_threshold = 0.3):

        n_samples -= diff_samples

        if self.layers_3D == 1:
            index = np.array([0])
        elif self.layers_3D == 3:
            index = np.array([-1,0,1])
        index += nn

        find_number = on_membrane_synapse
        on_synapse  = on_membrane_synapse

        offset = (self.in_window_shape[0]/2)
        temp = imarray[nn,offset:-(offset),offset:-(offset)]
        temp = temp[::sample_stride,::sample_stride]

        ix = np.in1d(temp.ravel(), find_number).reshape(temp.shape)
        temp_edge_x,temp_edge_y = np.where(ix)
        temp_edge_x.flags.writeable = True
        temp_edge_y.flags.writeable = True
        temp_edge_x *= sample_stride
        temp_edge_y *= sample_stride

        if temp_edge_x.size < n_samples:
            print 'Warning: Not enough samples...',temp_edge_x.size
            diff_samples = n_samples-temp_edge_x.size
        rand = np.random.permutation(range(temp_edge_x.size))
        rand = rand[:n_samples]

        edge_x = np.zeros(rand.size)
        edge_y = np.zeros(rand.size)
        for n in xrange(rand.size):
            edge_x[n] = temp_edge_x[rand[n]]
            edge_y[n] = temp_edge_y[rand[n]]

        x = np.zeros((rand.size,self.layers_3D*self.in_window_shape[0]*self.in_window_shape[1]))
        y = np.zeros((rand.size,self.out_window_shape[0]*self.out_window_shape[1]))

        for n in xrange(rand.size):

            in_start_point_x  = edge_x[n]
            in_end_point_x    = in_start_point_x + self.in_window_shape[0]
            in_start_point_y  = edge_y[n]
            in_end_point_y    = in_start_point_y + self.in_window_shape[1]
            out_start_point_x = edge_x[n] + (self.in_window_shape[0]-self.out_window_shape[0])/2
            out_end_point_x   = out_start_point_x + self.out_window_shape[0]
            out_start_point_y = edge_y[n] + (self.in_window_shape[1]-self.out_window_shape[1])/2
            out_end_point_y   = out_start_point_y + self.out_window_shape[1]

            edge_sample  = thick_edged[nn,out_start_point_x:out_end_point_x,out_start_point_y:out_end_point_y]
            image_sample = input_image[index,in_start_point_x:in_end_point_x,in_start_point_y:in_end_point_y]

            edge_sample  = edge_sample.reshape(self.out_window_shape[0]*self.out_window_shape[1],)
            image_sample = image_sample.reshape(self.layers_3D*self.in_window_shape[0]*self.in_window_shape[1],)

            x[n] = image_sample
            y[n] = edge_sample

        return x,y,diff_samples


    def sample_synapse_reg(self,nn,imarray,thick_edged,input_image,n_samples,image_group_train,sample_stride = 4,on_membrane_synapse = False,diff_samples = 0,on_synapse_threshold = 0.1):

        n_samples -= diff_samples

        if self.layers_3D == 1:
            index = np.array([0])
        elif self.layers_3D == 3:
            index = np.array([-1,0,1])
        index += nn

        find_number = on_membrane_synapse
        on_synapse  = on_membrane_synapse

        offset = (self.in_window_shape[0]-self.out_window_shape[0])/2
        temp = imarray[nn,offset:-(offset),offset:-(offset)]
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

        x = np.zeros((rand.size,self.layers_3D*self.in_window_shape[0]*self.in_window_shape[1]))
        y = np.zeros((rand.size,1))
        for n in xrange(rand.size):

            in_start_point_x = x_samples[n]
            in_end_point_x   = in_start_point_x + self.in_window_shape[0]
            in_start_point_y = y_samples[n]
            in_end_point_y   = in_start_point_y + self.in_window_shape[1]

            out_sample = temp[x_samples[n],y_samples[n]]
            image_sample = input_image[index,in_start_point_x:in_end_point_x,in_start_point_y:in_end_point_y]

            out_sample = out_sample.reshape(1,)
            image_sample = image_sample.reshape(self.layers_3D*self.in_window_shape[0]*self.in_window_shape[1],)

            x[n] = image_sample
            y[n] = out_sample

        return x,y,diff_samples

    def read_in_images(self,directory_input,directory_labels):
        ''' 
        Function that reads in images from file and do
        some pre-processing
        '''

        files_input  = []
        files_labels = []

        img_real_stack = np.zeros((0,self.img_size[0]**2))

        image_groups = [0]
        counter = 0
        for directory in directory_input:
            files_input = sorted(glob.glob(directory+"/*.tif"))
        
            for File in files_input:
                
                img_temp = Image.open(File)                                                        
                flag = True                                                                     
                i = 0                                                                           
                while flag == True:                                                             
                    try:                                                                        
                        img_temp.seek(i)                           
                        img_temp_temp = np.array(img_temp.getdata()).reshape(img_temp.size)                                  
                        img_real_stack = np.vstack((img_real_stack,img_temp_temp.flatten(1)))           
                        i += 1     
                        counter +=1
                    except EOFError:                                                            
                        flag = False 

            image_groups.append(counter)
	
        img_stack = np.zeros((0,self.img_size[0]**2))
	
        for directory in directory_labels:
            files_labels = sorted(glob.glob(directory+"/*.tif"))
            for File in files_labels:
                img_temp = Image.open(File)                                                        
                flag = True                                                                     
                i = 0                                                                         
                while flag == True:                                                             
                    try:            
                        img_temp.seek(i)
                        img_temp_temp = np.array(img_temp.getdata()).reshape(img_temp.size)
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
        if self.n_train_files == None:
            train_img_input  = img_real_stack[:(total_files-self.n_test_files)]
            train_img_labels = img_stack[:(total_files-self.n_test_files)]
        else:
            train_img_input  = img_real_stack[:self.n_train_files]
            train_img_labels = img_stack[:self.n_train_files]

        #Add starting and end point for image stacks
        image_groups_train = image_groups
        image_groups_train[-1] = total_files -self.n_test_files

        img_group_train = np.zeros((len(image_groups_train)-1,2),dtype=np.int32)
        n = 0
        for n in xrange(len(image_groups_train)-1):
            img_group_train[n,0] = image_groups_train[n]
            img_group_train[n,1] = image_groups_train[n+1]-1
            n +=1

        test_img_input  = img_real_stack[(total_files-self.n_test_files):]
        test_img_labels = img_stack[(total_files-self.n_test_files):]

        img_group_test = np.zeros(2,dtype=np.int32)
        img_group_test[0] = 0
        img_group_test[-1] = self.n_test_files -1

        train_img_input  = train_img_input.reshape(train_img_input.shape[0],self.img_size[0],self.img_size[1])
        train_img_labels = train_img_labels.reshape(train_img_labels.shape[0],self.img_size[0],self.img_size[1])
        test_img_input   = test_img_input.reshape(test_img_input.shape[0],self.img_size[0],self.img_size[1])
        test_img_labels  = test_img_labels.reshape(test_img_labels.shape[0],self.img_size[0],self.img_size[1])
	
        return train_img_input,train_img_labels,test_img_input,test_img_labels,img_group_train,img_group_test

    def process_images(self,train_img_input,train_img_labels):
        labeled_in  = np.zeros(train_img_labels.shape)
        labeled_out = np.zeros(train_img_labels.shape)

        n = 0
        for n in range(train_img_labels.shape[0]):

            if self.classifier == 'membrane':
                labeled_in[n] = train_img_labels[n]
                labeled_in[n] = labeled_in[n]/labeled_in[n].max()

                if self.membrane_edges == 'WideEdges':
                    labeled_out[n] = self.thick_edge(labeled_in[n])

                elif self.membrane_edges == 'GaussianBlur':
                    labeled_out[n] = scipy.ndimage.gaussian_filter(labeled_in[n], sigma=self.sigma)
                    labeled_out[n] = labeled_out[n]/labeled_out[n].max()
                else:
                    labeled_out[n] = labeled_in[n]
                    print "Warning: thin edge"

            elif self.classifier == 'synapse':
                labeled_in[n],labeled_out[n] = train_img_labels[n],train_img_labels[n]

            elif self.classifier == 'synapse_reg':
                labeled_in[n],labeled_out[n] = train_img_labels[n],train_img_labels[n]

            if self.adaptive_histogram_equalization:
                # Adaptive Equalization
                pass
                #train_img_input[n] = exposure.equalize_adapthist(train_img_input[n], clip_limit=0.03)

        return labeled_in,labeled_out,train_img_input

    def generate_test_membrane_synapse(self,thick_edged, img_real, img_number):
        
        # Define indexes for 1D and 3D 
        if self.layers_3D == 1:
            index = np.array([0])
        elif self.layers_3D == 3:
            index = np.array([-1,0,1])

        index += img_number

        # Generate test set for classes: (membrane,synapse)
        offset = self.in_window_shape[0]/2
        diff  = (self.in_window_shape[0]-self.out_window_shape[0])

        thick_edged = thick_edged[img_number,(diff/2):-(diff/2),(diff/2):-(diff/2)]
        
        number = thick_edged.shape[0]/self.stride - (self.out_window_shape[0]/self.stride - 1)

        img_samples = np.zeros((number**2,self.layers_3D*self.in_window_shape[0]**2))
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

                img_samples[table_number,:] = img_real[index,img_start_y:img_end_y,img_start_x:img_end_x].reshape(1,self.layers_3D*self.in_window_shape[0]**2)
                labels[table_number,:]      = thick_edged[label_start_y:label_end_y, label_start_x:label_end_x].reshape(1,self.out_window_shape[0]**2)

                table[table_number,0] = img_number
                table[table_number,1] = label_start_y
                table[table_number,2] = label_start_x
                table_number += 1

        return img_samples,labels,table

    def generate_test_synapse_reg(self,thick_edged, img_real, img_number):
        
        # Define indexes for 1D and 3D 
        if self.layers_3D == 1:
            index = np.array([0])
        elif self.layers_3D == 3:
            index = np.array([-1,0,1])

        index += img_number
        # Generate test set for classes: (synapse_reg)
        diff  = (self.in_window_shape[0]-self.out_window_shape[0])

        thick_edged = thick_edged[img_number,(diff/2):-(diff/2),(diff/2):-(diff/2)]
        
        number = thick_edged.shape[0]/self.out_window_shape[0]

        img_samples = np.zeros((number**2,self.layers_3D*self.in_window_shape[0]**2))
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

                img_samples[table_number,:] = img_real[index,img_start_y:img_end_y,img_start_x:img_end_x].reshape(self.layers_3D*self.in_window_shape[0]**2,)
                labels[table_number,0]      = np.mean(thick_edged[label_start_y:label_end_y, label_start_x:label_end_x])

                table[table_number,0] = img_number
                table[table_number,1] = label_start_y/float(self.out_window_shape[0])
                table[table_number,2] = label_start_x/float(self.out_window_shape[0])
                table_number += 1

        return img_samples,labels,table

    
    def generate_data(self, config_file):

        print('Loading images ...')

        # Load in images to numpy array(image_nr,x_dimension,y_dimension) and
        # table that defines the groups of images ((group1_start,
        # group1_end),(group2_start,group_end))
        train_img_input,train_img_labels,test_img_input,test_img_labels,img_group_train,img_group_test = self.read_in_images(self.directory_input,self.directory_labels)
        
        # Process train images, find synapses, edges and do edge processing
        # (blurring, widening) if specified
        labeled_in,labeled_out, train_img_input = self.process_images(train_img_input,train_img_labels)

        print('Pre-processing images...')

        # Define the fraction of samples on synapse/edge and off synapse edge.
        # By default set to 0.5
        on_samples  = int(self.samples_per_image*self.on_ratio)
        off_samples = self.samples_per_image - on_samples

        # Define training arrays and functions
        if self.classifier in ['membrane','synapse']:
            train_x = np.zeros((0,self.layers_3D*self.in_window_shape[0]**2))
            train_y = np.zeros((0,self.out_window_shape[0]**2))
            sample_function = self.sample_membrane_synapse
        elif self.classifier == 'synapse_reg':
            train_x = np.zeros((0,self.layers_3D*self.in_window_shape[0]**2))
            train_y = np.zeros((0,1))
            sample_function = self.sample_synapse_reg

        # Sample training data from training images
        for n in range(train_img_input.shape[0]):
                if n not in img_group_train or self.layers_3D == 1:
                    x_temp, y_temp = np.zeros((0, self.in_window_shape[0]*self.in_window_shape[1])), np.zeros((0, self.out_window_shape[0]*self.out_window_shape[1])) 
                    x_temp,y_temp, diff_samples = sample_function(n, labeled_in, labeled_out, train_img_input,on_samples, img_group_train,on_membrane_synapse = True)
                    
                    try:
                        train_x = np.vstack((train_x, x_temp))
                        train_y = np.vstack((train_y, y_temp))
                    except:
                        train_x = np.vstack((train_x, x_temp))
                        train_y = np.append(train_y, y_temp)

                    x_temp, y_temp = np.zeros((0, self.in_window_shape[0]*self.in_window_shape[1])),np.zeros((0, self.out_window_shape[0]*self.out_window_shape[1])) 
                    x_temp, y_temp,diff_samples = sample_function(n, labeled_in, labeled_out, train_img_input, off_samples, img_group_train, on_membrane_synapse = False, diff_samples=diff_samples)

                    try:
                        train_x = np.vstack((train_x, x_temp))
                        train_y = np.vstack((train_y, y_temp))
                    except:
                        train_x = np.vstack((train_x, x_temp))
                        train_y = np.append(train_y, y_temp)

        folder_name = 'pre_process/data_strucs/' + config_file
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        np.save(folder_name + '/x_train.npy',train_x)
        np.save(folder_name + '/y_train.npy',train_y)

        print "Finished train set"

        # Process test images (wide edges, add gaussian blur etc.)
        labeled_in,labeled_out,test_img_input = self.process_images(test_img_input,test_img_labels)

        # Define training arrays 
        if self.classifier in ['membrane','synapse']:
            test_x = np.zeros((0,self.layers_3D*self.in_window_shape[0]**2))
            test_y = np.zeros((0,self.out_window_shape[0]**2))
            generate_test_set = self.generate_test_membrane_synapse
        elif self.classifier == 'synapse_reg':
            test_x = np.zeros((0,self.layers_3D*self.in_window_shape[0]**2))
            test_y = np.zeros((0,1))
            generate_test_set = self.generate_test_synapse_reg

        table = np.zeros((0,3),dtype=np.int32)

        # Define test samples
        img_number = 0
        for n in range(test_img_input.shape[0]):
            if n not in img_group_test or self.layers_3D == 1:

                img_samples,labels,table_temp = generate_test_set(labeled_out,test_img_input,img_number)

                test_x = np.vstack((test_x,img_samples))
                test_y = np.vstack((test_y,labels))
                table = np.vstack((table,table_temp))
                
                img_number += 1

        np.save(folder_name + '/x_test.npy',test_x)
        np.save(folder_name + '/y_test.npy',test_y)
        np.save(folder_name + '/table.npy',table)
        
        print 'Done ... '

if __name__ == '__main__':
    read = Read()
   
    
    
