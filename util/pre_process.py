import numpy as np
import theano
import theano.tensor as T
from theano import shared, function
import os
theano.config.floatX = 'float32'
rng = np.random.RandomState(42)

class PreProcess(object):

    def __init__(self,n_val_samples):
        self.n_val_samples = n_val_samples
    
    def run(self):

        # Load training and test set 
        train_set_x = np.load('data/x_train.npy')
        train_set_y = np.load('data/y_train.npy')
        test_set_x  = np.load('data/x_test.npy')
        test_set_y  = np.load('data/y_test.npy')


        if train_set_y.ndim != 2 or test_set_y.ndim != 2:
            train_set_y = train_set_y.reshape(train_set_y.shape[0],1)
            test_set_y  = test_set_y.reshape(test_set_y.shape[0],1)

        print train_set_x.shape
        print test_set_x.shape
        print train_set_y.shape
        print test_set_y.shape

        valid_set_size = self.n_val_samples
        
        print 'Size of training/test-set: ',train_set_x.shape[0],'/',test_set_x.shape[0]
        
        rand_val = np.random.permutation(range(test_set_x.shape[0]))[:valid_set_size]
        valid_set_x = np.zeros((valid_set_size,train_set_x.shape[1]))
        valid_set_y = np.zeros((valid_set_size,train_set_y.shape[1]))
        for n in xrange(len(rand_val)):
            valid_set_x[n] = test_set_x[rand_val[n]]
            valid_set_y[n] = test_set_y[rand_val[n]]

        # Flip a number of the training data
        flip_prob = 0.5
        number_flips = np.int(np.floor(train_set_x.shape[0]*flip_prob))
        rand = np.random.permutation(range(train_set_x.shape[0]))[:number_flips]
        
        for n in xrange(rand.size):
            train_set_x[rand[n]] = train_set_x[rand[n],::-1]
            train_set_y[rand[n]] = train_set_y[rand[n],::-1]
            
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
    
