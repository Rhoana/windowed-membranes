import numpy as np
import glob
import os
from scipy import signal
import theano

from pre_process.images_from_file  import ImagesFromFile
from pre_process.process           import Process
from pre_process.sample            import Sample
from pre_process.generate_test_set import GenerateTestSet

import util.helper_functions as f 
import post_process.post_process as post 

rng              = np.random.RandomState(42)

class Worker(object):
    """
    Worker class to generate train/valid/test sets.
    """

    def __init__(self,in_window_shape,
            out_window_shape,
            pred_window_size,
            stride,
            img_size,
            classifier,
            n_train_files,
            n_test_files,
            samples_per_image,
            on_ratio,
            directory_input,
            directory_labels,
            membrane_edges,
            layers_3D,
            pre_processed_folder,
            batch_size,
            num_kernels,
            kernel_sizes,
            maxoutsize,
            params,
            eval_window_size,
            config_file,
            n_valid_samples):

        self.in_window_shape                 = in_window_shape
        self.out_window_shape                = out_window_shape
        self.pred_window_size                = pred_window_size
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
        self.pre_processed_folder            = pre_processed_folder
        self.batch_size                      = batch_size
        self.num_kernels                     = num_kernels
        self.kernel_sizes                    = kernel_sizes
        self.maxoutsize                      = maxoutsize
        self.params                          = params
        self.eval_window_size                = eval_window_size
        self.config_file                     = config_file
        self.n_valid_samples                 = n_valid_samples

        # Initialize generation of sets
        read_images = ImagesFromFile(self.n_train_files,self.n_test_files,self.img_size,self.classifier)
        (train_files_input,
                train_files_labeled,
                test_files_input,
                test_files_labeled,
                img_group_train,
                img_group_test) = read_images.init_train_test(self.directory_input, self.directory_labels)

        self.read_images         = read_images
        self.train_files_input   = train_files_input
        self.train_files_labeled = train_files_labeled
        self.test_files_input    = test_files_input
        self.test_files_labeled  = test_files_labeled
        self.img_group_train     = img_group_train
        self.img_group_test      = img_group_test
        
        self.n_test_samples = len(self.test_files_labeled)

        # Initialize process
        self.process = Process()
        # Initialize generation of test data
        self.init_test_data()

    def generate_train_data(self):
        """
        Generate training data and store it as .npy file.
        """

        print('Loading train images ...')

        # Load in images to numpy array(image_nr,x_dimension,y_dimension) and
        # table that defines the groups of images ((group1_start,
        # group1_end),(group2_start,group_end))

        train_img_input, train_img_labels = self.read_images.read_in_images(self.train_files_input,self.train_files_labeled)

        (train_img_input,
                labeled_in_train,
                labeled_out_train) = self.process.process_images(train_img_input,
                    train_img_labels,
                    self.classifier,
                    self.membrane_edges)

        sample = Sample(self.samples_per_image,
                self.on_ratio,
                self.layers_3D,
                self.classifier,
                self.in_window_shape,
                self.out_window_shape,
                self.img_group_train,
                self.config_file,
                self.pre_processed_folder)
        sample.run_sampling(labeled_in_train,labeled_out_train,train_img_input)
        
        
    def get_train_data(self,n_valid_images = 1):
        """
        Get training data and validation set and do
        normalization of the data.
        """

        try:
            train_set_x = np.load(self.pre_processed_folder + 'x_train.npy')
            train_set_y = np.load(self.pre_processed_folder + 'y_train.npy')
        except:
            print "Error: Unable to load pre-processed train set."
            exit()

        n_train_samples = train_set_x.shape[0]
        print 'Size of training-set: ',n_train_samples

        if train_set_y.ndim != 2:
            train_set_y = train_set_y.reshape(train_set_y.shape[0],1)
            
        valid_set_x,valid_set_y = self.generate_val_set(n_valid_images)

        # estimate the mean and std dev from the training data
        # then use these estimates to normalize the data
        # estimate the mean and std dev from the training data
    
        norm_mean = train_set_x.mean()
        norm_std = train_set_x.std()
        norm_std = norm_std.clip(0.00001, norm_std)

        train_set_x = train_set_x - norm_mean
        train_set_x = train_set_x / norm_std 

        valid_set_x = valid_set_x - norm_mean
        valid_set_x = valid_set_x / norm_std 
    
        train_set_x = train_set_x.astype(np.float32)
        valid_set_x = valid_set_x.astype(np.float32)

        train_set_y = train_set_y.astype(np.float32)
        valid_set_y = valid_set_y.astype(np.float32)

        train_set_x = theano.tensor._shared(train_set_x,borrow=True)
        valid_set_x = theano.tensor._shared(valid_set_x,borrow=True)

        train_set_y = theano.tensor._shared(train_set_y,borrow=True)
        valid_set_y = theano.tensor._shared(valid_set_y,borrow=True)
    
        list_it = [train_set_x,valid_set_x,train_set_y,valid_set_y]
    
        return list_it, n_train_samples
            
            
    def generate_test_data(self,model,x,y,index,net):
        """
        Generate and predict test data
        """

        finished = False

        error_pixel_before  = 0.
        error_window_before = 0.
        error_pixel_after   = 0.
        error_window_after  = 0.
        
        for n in xrange(999999999999999999):

            test_x, test_y,table, finished  = self.get_new_test_sample()
            
            if finished == True:
                break

            if n == 0:
                test_set_x  = theano.tensor._shared(test_x,borrow=True)             
                test_set_y  = theano.tensor._shared(test_y,borrow=True) 
            else:
                test_set_x.set_value(test_x)
                test_set_y.set_value(test_y)

            if n == 0:
                # Timer information
                number_test_pixels  = test_set_y.get_value(borrow=True).shape[0]*test_set_y.get_value(borrow=True).shape[1]
                
                # adjust batch size
                n_test_batches = test_set_x.get_value(borrow=True).shape[0]
                test_batch_size = self.batch_size
                while n_test_batches % test_batch_size != 0:
                    test_batch_size += 1 

                n_test_batches /= test_batch_size

                if net == 'ConvNet':
                  model_test = model.TestVersion(rng, 
                      test_batch_size, 
                      self.layers_3D, 
                      self.num_kernels, 
                      self.kernel_sizes, 
                      x, 
                      y,
                      self.in_window_shape,
                      self.out_window_shape,
                      self.pred_window_size,
                      self.classifier,
                      maxoutsize = self.maxoutsize, 
                      params = self.params, 
                      network = model, 
                      dropout = [0.,0.,0.,0.0])

                elif net == "FullyCon" or net == "FullyConCompressed":
                  model_test= model.TestVersion(rng, 
                      test_batch_size, 
                      self.layers_3D, 
                      x, 
                      y,
                      self.in_window_shape,
                      self.out_window_shape,
                      self.pred_window_size,
                      self.classifier,
                      params = self.params, 
                      network = model)          
                else:
                  raise RuntimeError('Unable to load network: ' + str(self.net))

            # Predict test set
            predict_test = f.init_predict(test_set_x,model_test,test_batch_size,x,index)
            predict_test = f.predict_set(predict_test, n_test_batches, self.classifier, self.pred_window_size, number_pixels=number_test_pixels)

            # Evaluate error
            error_pixel_before_t,error_window_before_t = f.evaluate(predict_test,test_set_y.get_value(borrow=True),self.eval_window_size,self.classifier)
            error_pixel_before  += error_pixel_before_t
            error_window_before += error_window_before_t

            predict_test, y_test = post.post_process(
                    predict_test,
                    test_set_y.get_value(borrow=True),
                    table,
                    self.img_size,
                    self.pred_window_size[0],
                    self.pred_window_size[1],
                    self.classifier)

            # Evalute error after post-processing 
            error_pixel_after_t,error_window_after_t = f.evaluate(predict_test,y_test,self.eval_window_size,self.classifier)
            error_pixel_after  += error_pixel_after_t
            error_window_after += error_window_after_t

            if n == 0:
                output = predict_test
                y      = y_test
                shape  = output.shape[2:]
                output = output.flatten(1)
                y      = y.flatten(1)
            else:
                output = np.vstack((output,predict_test.flatten(1)))
                y      = np.vstack((y, y_test.flatten(1)))

        
        self.end_test_gen()
        
        output = output.reshape(n, shape[0], shape[1])
        y      = y.reshape(n,shape[0],shape[1])

        error_pixel_before  /= float(n+1)
        error_window_before /= float(n+1)
        error_pixel_after   /= float(n+1)
        error_window_after  /= float(n+1)
        
        return output, y, error_pixel_before, error_window_before, error_pixel_after, error_window_after


    def init_test_data(self):

        print('Loading test images ...')

        test_img_input, test_img_labels = self.read_images.read_in_images(self.test_files_input,self.test_files_labeled)

        # Process test images (wide edges, add gaussian blur etc.)
        (test_img_input,
                labeled_in_test,
                labeled_out_test) = self.process.process_images(test_img_input,
                        test_img_labels,
                        self.classifier,
                        self.membrane_edges)

        self.gen_test_set = GenerateTestSet(self.pred_window_size[0],
                self.pred_window_size[1],
                self.layers_3D,
                self.classifier,
                self.stride,
                self.config_file,
                self.pre_processed_folder,
                self.test_files_labeled,
                test_img_input,
                labeled_out_test,
                self.img_group_test)

        self.image_number = 0
        self.n_test_samples = len(self.test_files_labeled)
        
    def generate_val_set(self,n_valid_samples):
        
        for n in xrange(n_valid_samples):
            img_number = np.random.randint(self.n_test_samples)
            test_x,test_y = self.get_valid_sample(img_number)
            try:
                test_set_x = np.vstack((test_set_x,test_x))
                test_set_y = np.vstack((test_set_y,test_y))
            except:
                test_set_x = test_x
                test_set_y = test_y
        
        return test_set_x,test_set_y
            
        
    def get_valid_sample(self,img_number):
        """
        Generate one new validation sample
        """
        test_set_x, test_set_y, table = self.gen_test_set.generate(img_number)
        return test_set_x,test_set_y

    def get_new_test_sample(self):
        """
        Generate one new test sample
        """
        if self.image_number < self.n_test_samples: 
            test_set_x, test_set_y, table = self.gen_test_set.generate(self.image_number)
            self.image_number += 1 
            finished = False
            
            if test_set_y.ndim != 2:
                test_set_y  = test_set_y.reshape(test_set_y.shape[0],1)

            norm_mean = test_set_x.mean()
            norm_std = test_set_x.std()
            norm_std = norm_std.clip(0.00001, norm_std)

            test_set_x = test_set_x - norm_mean
            test_set_x = test_set_x / norm_std 

            test_set_x = test_set_x.astype(np.float32)
            test_set_y = test_set_y.astype(np.float32)
        else:
            test_set_x  = None
            test_set_y  = None
            table       = None
            finished    = True


        return test_set_x, test_set_y, table, finished

    def end_test_gen(self):
        self.gen_test_set.end_test_generation()


if __name__ == '__main__':
    read = Read()
   
    
    
