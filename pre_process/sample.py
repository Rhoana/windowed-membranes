import numpy as np
import random
import os

class Sample(object):

    def __init__(self,samples_per_image,on_ratio,layers_3D,classifier,in_window_shape,out_window_shape,img_group_train,config_file,pre_processed_folder):
        self.layers_3D = layers_3D
        self.classifier = classifier
        self.in_window_shape = in_window_shape
        self.out_window_shape = out_window_shape
        self.img_group_train = img_group_train
        self.in_window_shape = in_window_shape
        self.out_window_shape = out_window_shape
        self.img_group_train = img_group_train
        self.pre_processed_folder = pre_processed_folder

        # Define the fraction of samples on synapse/edge and off synapse edge.
        # By default set to 0.5
        if self.classifier in ["membrane","synapse"]:
            self.sample1  = int(samples_per_image*on_ratio)
            self.sample0 = samples_per_image - self.sample1
        elif self.classifier == "membrane_synapse":
            self.sample0 = int(0.33*samples_per_image)
            self.sample1 = int(0.33*samples_per_image)
            self.sample2 = int(0.33*samples_per_image)

        self.config_file = config_file

        # Define training arrays and functions
        if self.classifier == "membrane" or self.classifier == "synapse":
            self.train_x = np.zeros((0,self.layers_3D*self.in_window_shape[0]**2))
            self.train_y = np.zeros((0,self.out_window_shape[0]**2))
            self.sample_function = self.sample_two_class
        else:
            self.classifier == "membrane_synapse"
            self.train_x = np.zeros((0,self.layers_3D*self.in_window_shape[0]**2))
            self.train_y = np.zeros((0,2*self.out_window_shape[0]**2))
            self.sample_function = self.sample_three_class

    def run_sampling(self,labeled_in,labeled_out,train_img_input):

        if self.classifier == "membrane_synapse":
            labeled_in = labeled_in[:,0] + labeled_in[:,1] *2
        else:
            labeled_in = labeled_in[:,0]
        # Sample training data from training images
        for n in range(train_img_input.shape[0]):
            if n not in self.img_group_train or self.layers_3D == 1:

                if self.classifier == "membrane_synapse":
                    x_temp, y_temp,diff_samples = self.sample_function(n, labeled_in, labeled_out, train_img_input, self.sample2, self.img_group_train, on_membrane_synapse = 2)

                    self.train_x = np.vstack((self.train_x, x_temp))
                    self.train_y = np.vstack((self.train_y, y_temp))
                else:
                    diff_samples = 0

                x_temp,y_temp, diff_samples = self.sample_function(n, labeled_in, labeled_out, train_img_input,self.sample1, self.img_group_train,on_membrane_synapse = 1, diff_samples = diff_samples)
                
                self.train_x = np.vstack((self.train_x, x_temp))
                self.train_y = np.vstack((self.train_y, y_temp))

                x_temp, y_temp,diff_samples = self.sample_function(n, labeled_in, labeled_out, train_img_input, self.sample0, self.img_group_train, on_membrane_synapse = 0, diff_samples=diff_samples)

                self.train_x = np.vstack((self.train_x, x_temp))
                self.train_y = np.vstack((self.train_y, y_temp))

        np.save(self.pre_processed_folder + 'x_train.npy',self.train_x)
        np.save(self.pre_processed_folder + 'y_train.npy',self.train_y)

        print "Finished train set"

    def sample_two_class(self,nn,imarray,thick_edged,input_image,n_samples,image_group_train,sample_stride = 4,on_membrane_synapse = False,diff_samples = 0):

        n_samples -= diff_samples

        if self.layers_3D == 1:
            index = np.array([0])
        elif self.layers_3D == 3:
            index = np.array([-1,0,1])
        index += nn

        find_number = on_membrane_synapse

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

            edge_sample  = thick_edged[nn,:,out_start_point_x:out_end_point_x,out_start_point_y:out_end_point_y]
            image_sample = input_image[index,in_start_point_x:in_end_point_x,in_start_point_y:in_end_point_y]

            edge_sample  = edge_sample.reshape(self.out_window_shape[0]*self.out_window_shape[1],)
            image_sample = image_sample.reshape(self.layers_3D*self.in_window_shape[0]*self.in_window_shape[1],)

            x[n] = image_sample
            y[n] = edge_sample

        return x,y,diff_samples

    def sample_three_class(self,nn,imarray,thick_edged,input_image,n_samples,image_group_train,sample_stride = 4,on_membrane_synapse = False,diff_samples = 0):

        n_samples -= diff_samples

        if self.layers_3D == 1:
            index = np.array([0])
        elif self.layers_3D == 3:
            index = np.array([-1,0,1])
        index += nn

        find_number = on_membrane_synapse

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
        y = np.zeros((rand.size,2*self.out_window_shape[0]*self.out_window_shape[1]))

        for n in xrange(rand.size):

            in_start_point_x  = edge_x[n]
            in_end_point_x    = in_start_point_x + self.in_window_shape[0]
            in_start_point_y  = edge_y[n]
            in_end_point_y    = in_start_point_y + self.in_window_shape[1]
            out_start_point_x = edge_x[n] + (self.in_window_shape[0]-self.out_window_shape[0])/2
            out_end_point_x   = out_start_point_x + self.out_window_shape[0]
            out_start_point_y = edge_y[n] + (self.in_window_shape[1]-self.out_window_shape[1])/2
            out_end_point_y   = out_start_point_y + self.out_window_shape[1]

            edge_sample  = thick_edged[nn,:,out_start_point_x:out_end_point_x,out_start_point_y:out_end_point_y]
            image_sample = input_image[index,in_start_point_x:in_end_point_x,in_start_point_y:in_end_point_y]

            edge_sample  = edge_sample.reshape(2*self.out_window_shape[0]*self.out_window_shape[1],)
            image_sample = image_sample.reshape(self.layers_3D*self.in_window_shape[0]*self.in_window_shape[1],)

            x[n] = image_sample
            y[n] = edge_sample

        return x,y,diff_samples
