import numpy as np
import glob
from PIL import Image
import PIL
from scipy import signal


class ImagesFromFile(object):
    """
    Class that read input/labeled images from file and generates
    input and training labels as numpy arrays. 
    """

    def __init__(self,n_train_files,n_test_files,img_size,classifier):
        self.n_train_files = n_train_files
        self.n_test_files  = n_test_files
        self.img_size      = img_size
        self.classifier    = classifier
        
    def convert_binary(self,imarray):
        """
        Convert everything greater than 0 to 1. 
        """
        imarray[imarray>0] = 1
        return imarray

    def edge_filter(self,img):
        """
        Find edge gradients
        """
        scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],[-10+0j, 0+ 0j, +10 +0j],[ -3+3j, 0+10j,  +3 +3j]])
        grad = signal.convolve2d(img, scharr, boundary='symm', mode='same')                                                    
        grad = np.absolute(grad).astype(np.float32)
        grad = grad/np.max(grad)                             
        return grad    

    def find_edges(self,img,threshold = 399):
        """
        Filter out synapses, find edges and 
        represent edges as binary.
        """
        img_out = img.copy()
        img_out[img_out>threshold] = 0

        img_out = self.edge_filter(img_out)
        edged = self.convert_binary(img_out)

        return edged

    def find_synapse(self,img,File,edges = False):
        """
        Filter out edges, find synapses and
        represent synapses as binary
        """
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
        """
        Generate representation for both membrane
        and synapse
        """

        img_syn = self.find_synapse(img,File = file)
        img_mem = self.find_edges(img)

        img = np.zeros((2,img_mem.shape[0],img_mem.shape[1]))
        img[0] = img_mem
        img[1] = img_syn

        return img

    def sort_key(self,input_file):   
        """
        Key for sorting the input/labeled images
        """                                                    
        return int(input_file.split('.')[0].split('_')[-1])  

    def init_train_test(self,directory_input,directory_labels):
        ''' 
        Initialize train and test files.
        
        Generate:
        - train/test_files_input
        - train/test_files_labeled
        - img_group_train/test - List over images at end of stacks
                                 that are not predicted during 3-layer
                                 prediction
        '''

        files_input  = []
        files_labels = []

        image_groups = [0]
        counter = 0

        total_files = 0
        input_files = []
        img_group = []
        for directory in directory_input:
            files = glob.glob(directory+"/*.tif") + glob.glob(directory+"/*.png")
            directory_files = sorted(files,key=self.sort_key)
            img_group.append(len(input_files))
            input_files += directory_files
            img_group.append(len(input_files)-1)
        labeled_files = []
        for directory in directory_labels:
            files = glob.glob(directory+"/*.tif") + glob.glob(directory+"/*.png")
            labeled_files += sorted(files,key=self.sort_key)

        print len(input_files)
        print len(labeled_files)

        total_files = len(input_files)
        train_files_input   = input_files[:-self.n_test_files]
        train_files_labeled = labeled_files[:-self.n_test_files]
        
        ##########################################################################
        # TEMP
        ##########################################################################
        predict_stack = True
        if predict_stack == False:
            test_files_input    = input_files[-self.n_test_files:]
            test_files_labeled  = labeled_files[-self.n_test_files:]
        else:
            test_files_input   = input_files
            test_files_labeled = labeled_files
        ########################################################################## 

        img_group.append(total_files-self.n_test_files-1)
        img_group.append(total_files-self.n_test_files)
        img_group = list(set(img_group))
        img_group_train = np.array(img_group)
        img_group_test  = np.array(img_group)-(total_files-self.n_test_files)
        img_group_train = img_group_train[img_group_train < (total_files-self.n_test_files)]
        img_group_test  = img_group_test[img_group_test>=0]

        return train_files_input, train_files_labeled, test_files_input, test_files_labeled, img_group_train, img_group_test, 
        
    def read_in_images(self,files_input, files_labeled):
        """
        Read in images and generate train/test set. 
        """
        
        post_process = True
        
        if post_process == False:
            img_input = np.zeros((len(files_input),self.img_size[0]**2))
            if self.classifier == "membrane_synapse":
                img_labels = np.zeros((len(files_labeled),2*self.img_size[0]**2))
            else:
                img_labels = np.zeros((len(files_labeled),self.img_size[0]**2))

            for n in xrange(len(files_input)):
                File = files_input[n]
                img_temp = Image.open(File)                                                        
                img_temp = np.array(img_temp).ravel()                                 
                img_input[n] = img_temp  

                File = files_labeled[n]
                img_temp = Image.open(File)                                                        
                img_temp = np.array(img_temp.getdata()).reshape(img_temp.size)
                img_temp.flags.writeable = True

                if self.classifier == 'membrane':
                    img_temp = self.find_edges(img_temp)
                elif self.classifier == 'synapse':
                    img_temp = self.find_synapse(img_temp,File)
                elif self.classifier == 'membrane_synapse':
                    img_temp = self.find_mem_syn(img_temp,File)
            

            if self.classifier == "membrane" or self.classifier == "synapse":
                output_dim = 1
            elif self.classifier == "membrane_synapse":
                output_dim = 2
            else:
                print "Error: invalid classifier"
                exit()

            img_input  = img_input.reshape(img_input.shape[0], self.img_size[0],self.img_size[1])
            img_labels = img_labels.reshape(img_labels.shape[0], output_dim, self.img_size[0],self.img_size[1])
        
        else:
            img_input = np.zeros((len(files_input),self.img_size[0]**2))
            if self.classifier == "membrane_synapse":
                img_labels = np.zeros((len(files_labeled),2*self.img_size[0]**2))
            else:
                img_labels = np.zeros((len(files_labeled),self.img_size[0]**2))
                
            for n in xrange(len(files_input)):
                File = files_input[n]
                img_temp = Image.open(File)  
                
                if n == 0:
                    img_size_input = np.array(img_temp).shape[0]
                    img_size_input = (img_size_input,img_size_input)
                    gap = (self.img_size[0]-img_size_input[0])/2
                    
                img_temp = img_temp.resize((self.img_size[0], self.img_size[1]), PIL.Image.ANTIALIAS)                                                      
                img_temp = np.array(img_temp).T.ravel() 
                                            
                img_input[n] = img_temp  

                File = files_labeled[n]
                img_temp = Image.open(File)
                
                if gap>0:
                    img_temp = np.array(img_temp)[gap:-gap,gap:-gap]
                    img_temp = Image.fromarray(np.uint8(img_temp))
                    img_temp = img_temp.resize((self.img_size[0], self.img_size[1]), PIL.Image.ANTIALIAS)
                                                                        
                img_temp = np.array(img_temp.getdata()).reshape(img_temp.size)
                img_temp.flags.writeable = True

                if self.classifier == 'membrane':
                    img_temp = self.find_edges(img_temp)
                elif self.classifier == 'synapse':
                    img_temp = self.find_synapse(img_temp,File)
                elif self.classifier == 'membrane_synapse':
                    img_temp = self.find_mem_syn(img_temp,File)

                img_labels[n] = img_temp.ravel()
            
                #import matplotlib.pyplot as plt
                #print img_input.shape
                #plt.figure()
                #plt.imshow(img_input[0].reshape(1024,1024))
                #plt.figure()
                #plt.imshow(img_labels[0].reshape(1024,1024))
                #plt.show()
                #exit()
            

            if self.classifier == "membrane" or self.classifier == "synapse":
                output_dim = 1
            elif self.classifier == "membrane_synapse":
                output_dim = 2
            else:
                print "Error: invalid classifier"
                exit()
                
            img_input  = img_input.reshape(img_input.shape[0], self.img_size[0],self.img_size[1])
            img_labels = img_labels.reshape(img_labels.shape[0], output_dim, self.img_size[0],self.img_size[1])

        return img_input, img_labels

