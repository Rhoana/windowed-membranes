import numpy as np
import theano
import theano.tensor as T
from theano import shared, function
import os
theano.config.floatX = 'float32'
rng = np.random.RandomState(42)

class BuildTrainTestSet(object):

    def __init__(self,n_val_samples,pre_processed_folder):
        self.n_val_samples = n_val_samples
        self.pre_processed_folder = pre_processed_folder
    
    def run(self, classifier,config_file):
        
<<<<<<< HEAD
        # Load training and test set 
        train_set_x = np.load(self.pre_processed_folder + 'x_train.npy')
        train_set_y = np.load(self.pre_processed_folder + 'y_train.npy')
        test_set_x  = np.load(self.pre_processed_folder + 'x_test.npy')
        test_set_y  = np.load(self.pre_processed_folder + 'y_test.npy')
=======

        folder_name = 'pre_process/data_strucs/' + config_file
        if not os.path.exists(folder_name):
            print "You must must pre-process this configuration first."

        # Load training and test set 
        train_set_x = np.load(folder_name + '/x_train.npy')
        train_set_y = np.load(folder_name + '/y_train.npy')
        test_set_x  = np.load(folder_name + '/x_test.npy')
        test_set_y  = np.load(folder_name + '/y_test.npy')
>>>>>>> 8551c69399b72a8c7005449f7d94ce78db48600d

        if train_set_y.ndim != 2 or test_set_y.ndim != 2:
            train_set_y = train_set_y.reshape(train_set_y.shape[0],1)
            test_set_y  = test_set_y.reshape(test_set_y.shape[0],1)

        valid_set_size = self.n_val_samples
        
        print 'Size of training/test-set: ',train_set_x.shape[0],'/',test_set_x.shape[0]
        
        rand_val = np.random.permutation(range(test_set_x.shape[0]))[:valid_set_size]
        valid_set_x = np.zeros((valid_set_size,train_set_x.shape[1]))
        valid_set_y = np.zeros((valid_set_size,train_set_y.shape[1]))
        for n in xrange(len(rand_val)):
            valid_set_x[n] = test_set_x[rand_val[n]]
            valid_set_y[n] = test_set_y[rand_val[n]]

        # estimate the mean and std dev from the training data
        # then use these estimates to normalize the data
        # estimate the mean and std dev from the training data
        
        norm_mean = train_set_x.mean()
        train_set_x = train_set_x - norm_mean
        norm_std = train_set_x.std()
        norm_std = norm_std.clip(0.00001, norm_std)
        train_set_x = train_set_x / norm_std 

        test_set_x = test_set_x - norm_mean
        test_set_x = test_set_x / norm_std 
        valid_set_x = valid_set_x - norm_mean
        valid_set_x = valid_set_x / norm_std 
        
        train_set_x = train_set_x.astype(np.float32)
        test_set_x = test_set_x.astype(np.float32)
        valid_set_x = valid_set_x.astype(np.float32)

        train_set_y = train_set_y.astype(np.float32)
        test_set_y = test_set_y.astype(np.float32)
        valid_set_y = valid_set_y.astype(np.float32)

        train_set_x = theano.tensor._shared(train_set_x,borrow=True)
        valid_set_x = theano.tensor._shared(valid_set_x,borrow=True)
        test_set_x  = theano.tensor._shared(test_set_x,borrow=True)

        train_set_y = theano.tensor._shared(train_set_y,borrow=True)
        valid_set_y = theano.tensor._shared(valid_set_y,borrow=True)
        test_set_y  = theano.tensor._shared(test_set_y,borrow=True)
        
        list_it = [train_set_x,valid_set_x,test_set_x,train_set_y,valid_set_y,test_set_y]
        
        return list_it

if __name__ == "__main__":
    pre_process =  PreProcess()
    pre_process.run()
    
