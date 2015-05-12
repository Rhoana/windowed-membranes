import numpy as np

class GenerateTestSet(object):

    def __init__(self,in_window_shape,out_window_shape,layers_3D,classifier,stride,config_file,pre_processed_folder):
        self.in_window_shape = in_window_shape
        self.out_window_shape = out_window_shape
        self.layers_3D = layers_3D
        self.classifier = classifier 
        self.stride = stride
        self.config_file = config_file
        self.pre_processed_folder = pre_processed_folder

    def generate(self,test_img_input,labeled_in,labeled_out,img_group_test):
        
        # Define training arrays 
        test_x = np.zeros((0,self.layers_3D*self.in_window_shape[0]**2))
        test_y = np.zeros((0,self.out_window_shape[0]**2))

        table = np.zeros((0,3),dtype=np.int32)

        # Define test samples
        img_number = 0
        for n in range(test_img_input.shape[0]):
            try:
                if n not in img_group_test or self.layers_3D == 1:

                    img_samples,labels,table_temp = self.generate_test_membrane_synapse(labeled_out,test_img_input,img_number)

                    test_x = np.vstack((test_x,img_samples))
                    test_y = np.vstack((test_y,labels))
                    table = np.vstack((table,table_temp))
                    
                    img_number += 1
            except MemoryError:
                print "Warning: Memory error, unable to process all images in test set."
                break

        np.save(self.pre_processed_folder + 'x_test.npy',test_x)
        np.save(self.pre_processed_folder + 'y_test.npy',test_y)
        np.save(self.pre_processed_folder + 'table.npy',table)

        print 'Done ... '

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

