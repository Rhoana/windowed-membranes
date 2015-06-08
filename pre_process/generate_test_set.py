import numpy as np
import cPickle

class GenerateTestSet(object):

    def __init__(self,in_window_shape,out_window_shape,layers_3D,classifier,stride,config_file,pre_processed_folder, test_address, test_img_input, labeled_out, img_group_test):
        self.in_window_shape = in_window_shape
        self.out_window_shape = out_window_shape
        self.layers_3D = layers_3D
        self.classifier = classifier 
        self.stride = stride
        self.config_file = config_file
        self.pre_processed_folder = pre_processed_folder
        self.test_address = test_address

        self.test_img_input = test_img_input
        self.labeled_out = labeled_out
        self.img_group_test = img_group_test 

    def generate(self,img_number):
        
        if img_number not in self.img_group_test or self.layers_3D == 1:
            img_samples,labels,table = self.generate_test_sample(self.labeled_out,self.test_img_input,img_number)

        else:
            del self.test_address[image_number]

        return img_samples, labels, table

    def generate_test_sample(self,thick_edged, img_real, img_number):

        
        # Define indexes for 1D and 3D 
        if self.layers_3D == 1:
            index = np.array([0])
        elif self.layers_3D == 3:
            index = np.array([-1,0,1])

        index += img_number

        # Generate test set for classes: (membrane,synapse)
        offset = self.in_window_shape/2
        diff  = (self.in_window_shape-self.out_window_shape)

        thick_edged = thick_edged[img_number,:,(diff/2):-(diff/2),(diff/2):-(diff/2)]
        
        number = thick_edged.shape[1]/self.stride - (self.out_window_shape/self.stride - 1)


        img_samples = np.zeros((number**2,self.layers_3D*self.in_window_shape**2))
        table = np.zeros((number**2,3),dtype=np.int32)

        if self.classifier == "membrane" or self.classifier == "synapse":
            out_dim = self.out_window_shape**2
            labels = np.zeros((number**2,out_dim))
            lab_dim = 1
        elif self.classifier == "membrane_synapse":
            out_dim = 2*self.out_window_shape**2
            labels = np.zeros((number**2,out_dim))
            lab_dim = 2

        table_number = 0
        for n in xrange(number):
            for m in xrange(number):

                img_start_y = self.stride*n
                img_end_y   = self.stride*n + self.in_window_shape
                img_start_x = self.stride*m
                img_end_x   = self.stride*m + self.in_window_shape

                label_start_y = self.stride*n
                label_end_y   = self.stride*n + self.out_window_shape
                label_start_x = self.stride*m
                label_end_x   = self.stride*m + self.out_window_shape

                img_samples[table_number,:] = img_real[index,img_start_y:img_end_y,img_start_x:img_end_x].reshape(1,self.layers_3D*self.in_window_shape**2)
                labels[table_number,:]      = thick_edged[:,label_start_y:label_end_y, label_start_x:label_end_x].reshape(1,lab_dim*self.out_window_shape**2)
                

                table[table_number,0] = 0
                table[table_number,1] = label_start_y
                table[table_number,2] = label_start_x
                table_number += 1

        return img_samples,labels,table

    def end_test_generation(self):
        
        f = file(self.pre_processed_folder +  'test_adress.dat', 'wb')
        cPickle.dump(self.test_address,f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

