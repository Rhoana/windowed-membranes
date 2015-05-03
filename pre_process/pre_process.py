#File for generating samples from input data
##
import numpy as np
import glob
import os
from scipy import signal
from images_from_file import ImagesFromFile
from process import Process
from sample import Sample
from generate_test_set import GenerateTestSet

class Read(object):

    def __init__(self,in_window_shape, out_window_shape, stride, img_size, classifier, n_train_files, n_test_files, samples_per_image, on_ratio, directory_input, directory_labels, membrane_edges,layers_3D, adaptive_histogram_equalization,pre_processed_folder,predict_only,predict_train_set,images_from_numpy):
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
        self.layers_3D                       = layers_3D
        self.adaptive_histogram_equalization = adaptive_histogram_equalization
        self.pre_processed_folder            = pre_processed_folder

        self.predict_only = predict_only
        self.predict_train_set = predict_train_set
        self.images_from_numpy = images_from_numpy

    def generate_data(self, config_file):

        print('Loading images ...')

        # Load in images to numpy array(image_nr,x_dimension,y_dimension) and
        # table that defines the groups of images ((group1_start,
        # group1_end),(group2_start,group_end))
        read_images = ImagesFromFile(self.n_train_files,self.n_test_files,self.img_size,self.classifier)
        if self.images_from_numpy == False:
            train_img_input,train_img_labels,test_img_input,test_img_labels,img_group_train,img_group_test = read_images.read_in_images(self.directory_input,self.directory_labels,self.predict_train_set)
        else:
            train_img_input,train_img_labels,test_img_input,test_img_labels,img_group_train,img_group_test = read_images.images_from_numpy(self.directory_input,self.directory_labels,self.predict_train_set)

        # Process train images, find synapses, edges and do edge processing
        # (blurring, widening) if specified
        process = Process()
        if self.predict_only == False:
            train_img_input,labeled_in_train,labeled_out_train = process.process_images(train_img_input,train_img_labels,self.classifier,self.membrane_edges,self.adaptive_histogram_equalization)

            sample = Sample(self.samples_per_image,self.on_ratio,self.layers_3D,self.classifier,self.in_window_shape,self.out_window_shape,img_group_train,config_file,self.pre_processed_folder)
            sample.run_sampling(labeled_in_train,labeled_out_train,train_img_input)

        # Process test images (wide edges, add gaussian blur etc.)
        test_img_input,labeled_in_test,labeled_out_test = process.process_images(test_img_input,test_img_labels,self.classifier,self.membrane_edges,self.adaptive_histogram_equalization)

        gen_test_set = GenerateTestSet(self.in_window_shape,self.out_window_shape,self.layers_3D,self.classifier,self.stride,config_file,self.pre_processed_folder)
        gen_test_set.generate(test_img_input,labeled_in_test,labeled_out_test,img_group_test)

if __name__ == '__main__':
    read = Read()
   
    
    
