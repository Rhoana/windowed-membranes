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
    
    def build_train_val_set(self):
        
        try:
            test_set_x  = np.load(self.pre_processed_folder + 'x_test.npy')
            test_set_y  = np.load(self.pre_processed_folder + 'y_test.npy')
        except:
            print "Error: Unable to load pre-processed test set."
            exit()

        if test_set_y.ndim != 2:
            test_set_y  = test_set_y.reshape(test_set_y.shape[0],1)

        valid_set_size = self.n_val_samples
        
        
        rand_val = np.random.permutation(range(test_set_x.shape[0]))[:valid_set_size]
        valid_set_x = np.zeros((valid_set_size,test_set_x.shape[1]))
        valid_set_y = np.zeros((valid_set_size,test_set_y.shape[1]))
        for n in xrange(len(rand_val)):
            valid_set_x[n] = test_set_x[rand_val[n]]
            valid_set_y[n] = test_set_y[rand_val[n]]

        try:
            train_set_x = np.load(self.pre_processed_folder + 'x_train.npy')
            train_set_y = np.load(self.pre_processed_folder + 'y_train.npy')
        except:
            print "Error: Unable to load pre-processed train set."
            exit()

        print 'Size of training-set: ',train_set_x.shape[0]

        if train_set_y.ndim != 2:
            train_set_y = train_set_y.reshape(train_set_y.shape[0],1)

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
        
        return list_it

    def build_test_set(self):
        try:
            test_set_x  = np.load(self.pre_processed_folder + 'x_test.npy')
            test_set_y  = np.load(self.pre_processed_folder + 'y_test.npy')
        except:
            print "Error: Unable to load pre-processed test set."
            exit()

        print "Size of test-set:", test_set_x.shape[0]

        if test_set_y.ndim != 2:
            test_set_y  = test_set_y.reshape(test_set_y.shape[0],1)

        norm_mean = test_set_x.mean()
        norm_std = test_set_x.std()
        norm_std = norm_std.clip(0.00001, norm_std)

        test_set_x = test_set_x - norm_mean
        test_set_x = test_set_x / norm_std 

        test_set_x = test_set_x.astype(np.float32)
        test_set_y = test_set_y.astype(np.float32)

        test_set_x  = theano.tensor._shared(test_set_x,borrow=True)
        test_set_y  = theano.tensor._shared(test_set_y,borrow=True)

        list_it = [test_set_x,test_set_y]
        return list_it
if __name__ == "__main__":
    pre_process =  PreProcess()
    pre_process.run()
    
