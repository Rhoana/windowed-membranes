import numpy as np
import glob
from PIL import Image
from scipy import signal


class ImagesFromFile(object):

    def __init__(self,n_train_files,n_test_files,img_size,classifier):
        self.n_train_files = n_train_files
        self.n_test_files = n_test_files
        self.img_size = img_size
        self.classifier = classifier

    def edge_filter(self,img):
        scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],[-10+0j, 0+ 0j, +10 +0j],[ -3+3j, 0+10j,  +3 +3j]])
        grad = signal.convolve2d(img, scharr, boundary='symm', mode='same')                                                    
        grad = np.absolute(grad).astype(np.float32)
        grad = grad/np.max(grad)                             
        return grad    

    def find_edges(self,img):

        threshold = 399
        # Remove synapses
        img_out = img.copy()
        for n in xrange(img.shape[0]):
            for m in xrange(img.shape[1]):
                if img[n,m] > threshold:
                    img_out[n,m] = 0

        img_out = self.edge_filter(img_out)

        edged = self.convert_binary(img_out)

        return edged

    def find_synapse(self,img,File,edges = False):

        if 'train-labels' in File:
            threshold = 399
        else:
            threshold = 0

        # Find synapse
        img_out = img.copy()
        for n in xrange(img.shape[0]):
            for m in xrange(img.shape[1]):
                if img[n,m] > threshold:
                    img_out[n,m] = 1
                else:
                    img_out[n,m] = 0

        return img_out

    def find_mem_syn(self,img,file):

        img_syn = self.find_synapse(img,File = file)
        img_mem = self.find_edges(img)

        img = np.zeros((2,img_mem.shape[0],img_mem.shape[1]))
        img[0] = img_mem
        img[1] = img_syn

        return img

    def convert_binary(self,imarray):
        imarray[np.where(imarray>0)] = 1
        return imarray

    def sort_key(self,input_file):                                                       
        return int(input_file.split('.')[0].split('_')[-1])  

    def read_in_images(self,directory_input,directory_labels,predict_train_set):
        ''' 
        Function that reads in images from file and do
        some pre-processing
        '''

        files_input  = []
        files_labels = []

        image_groups = [0]
        counter = 0

        img_real_stack = np.zeros((0,self.img_size[0]**2))
        for directory in directory_input:
            files_input = sorted(glob.glob(directory+"/*.tif"),key=self.sort_key)
            if len(files_input) == 0:
                print "Error: Unable to read directory:",directory
                exit()
        
            for File in files_input:

                img_temp = Image.open(File)                                                        
                img_temp = np.array(img_temp.getdata()).flatten(1)                                  
                img_real_stack = np.vstack((img_real_stack,img_temp))           
                counter +=1

            image_groups.append(counter)

        if self.classifier == "membrane_synapse":
            img_stack = np.zeros((0,2*self.img_size[0]**2))
        else:
            img_stack = np.zeros((0,self.img_size[0]**2))
	
        address_list = []
        n = 0
        for directory in directory_labels:
            files_labels = sorted(glob.glob(directory+"/*.tif"),key=self.sort_key)
            if len(files_labels) == 0:
                print "Error: Unable to read directory:",directory
                exit()

            for File in files_labels:
                address_list.append(File)
                img_temp = Image.open(File)                                                        
                img_temp = np.array(img_temp.getdata()).reshape(img_temp.size)
                img_temp.flags.writeable = True
                if self.classifier == 'membrane':
                    img_temp = self.find_edges(img_temp)
                elif self.classifier == 'synapse':
                    img_temp = self.find_synapse(img_temp,File)
                elif self.classifier == 'membrane_synapse':
                    img_temp = self.find_mem_syn(img_temp,File)

                img_stack = np.vstack((img_stack,img_temp.ravel()))  

        
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
        test_address    = address_list[(total_files-self.n_test_files):]

        img_group_test = np.zeros(2,dtype=np.int32)
        img_group_test[0] = 0
        img_group_test[-1] = self.n_test_files -1

        if self.classifier == "membrane" or self.classifier == "synapse":
            output_dim = 1
        elif self.classifier == "membrane_synapse":
            output_dim = 2
        else:
            print "Error: invalid classifier"
            exit()

        train_img_input  = train_img_input.reshape(train_img_input.shape[0], self.img_size[0],self.img_size[1])
        train_img_labels = train_img_labels.reshape(train_img_labels.shape[0], output_dim, self.img_size[0],self.img_size[1])
        test_img_input   = test_img_input.reshape(test_img_input.shape[0], self.img_size[0],self.img_size[1])
        test_img_labels  = test_img_labels.reshape(test_img_labels.shape[0],output_dim, self.img_size[0],self.img_size[1])

        return train_img_input,train_img_labels,test_img_input,test_img_labels,img_group_train,img_group_test,test_address

    def images_from_numpy(self,train_file_x,train_file_y,test_file_x,test_file_y,img_group_train,img_group_test):

        if self.predict_train_set != True:
            train_x = np.load(train_file_x)
            train_y = np.load(train_file_y)
            test_x  = np.load(test_file_x)
            test_y  = np.load(test_file_y)
        else:
            train_x = np.load(train_file_x)
            train_y = np.load(train_file_y)
            test_x = train_x
            test_y = train_y
	
        return train_img_input,train_img_labels,test_img_input,test_img_labels,img_group_train,img_group_test
