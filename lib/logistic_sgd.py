import os
import sys
import time
import numpy
import theano
import theano.tensor as T

class LogisticRegression(object):
    """
    Logistic regression class
    """

    def __init__(self, input, n_in, n_out, out_window_shape, classes = 2, W = None, b = None,
            params = {}, params_number = None, classifier = 'standard'):

        self.out_window_shape = out_window_shape
        self.classifier = classifier

        if W == None or b == None:
            W_name = "W" + str(params_number)                                         
            b_name = "b" + str(params_number)                                         
                                                                                    
            if params.has_key(W_name) and params.has_key(b_name):  
                W = theano.shared(
                        params[W_name],
                    name= W_name,
                    borrow=True
                )
                # Initialize biases
                b = theano.shared(
                    params[b_name],
                    name= b_name,
                    borrow=True
                )

            else:
                # Initialize weights
                W = theano.shared(
                    value=numpy.zeros(
                        (n_in, n_out),
                        dtype=theano.config.floatX
                    ),
                    name=W_name,
                    borrow=True
                )
                # Initialize biases
                b = theano.shared(
                    value=numpy.zeros(
                        (n_out),
                        dtype=theano.config.floatX
                    ),
                    name=b_name,
                    borrow=True
                )

        self.W = W
        self.b = b
        
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        # Define parameters in list
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y, penatly_factor):
        '''
        Return cost function
        '''
        # Calculate sum of derivatives
        test = self.p_y_given_x.reshape((self.p_y_given_x.shape[0],self.out_window_shape[0],self.out_window_shape[1]))
        test_dx = (test[:,1:,:] - test[:,:-1,:]).reshape((self.p_y_given_x.shape[0],(self.out_window_shape[0]-1)*self.out_window_shape[1]))
        test_dy = (test[:,:,1:] - test[:,:,:-1]).reshape((self.p_y_given_x.shape[0],(self.out_window_shape[0]-1)*self.out_window_shape[1]))
        term = T.mean(T.abs_(test_dx)+T.abs_(test_dy),axis=1)
        
        # Calculate cost function
        if self.classifier in ['membrane','synapse']:
            L = - T.mean( y* T.log(self.p_y_given_x) + (1 - y) * T.log(1 - self.p_y_given_x), axis=1)
            L = T.mean(L+term*penatly_factor)
        elif self.classifier == 'synapse_reg':
            L = T.mean((y-self.p_y_given_x)**2)
        return L

    def errors(self, y):
        '''
        Output errors
        '''
        prediction = self.p_y_given_x
        L = T.mean(T.abs_(prediction-y),axis=1)
        return T.mean(L)
    
    def prediction(self):
        '''
        Output prediction
        '''
        return self.p_y_given_x

    def TestVersion(self,input, n_in, n_out, out_window_shape, classes = 2, W = None, b = None,
            params = {}, params_number = None, classifier = 'standard'):
        return LogisticRegression(input, n_in, n_out, out_window_shape, classes = classes, W = self.W, b = self.b,
            params = params, params_number = params_number, classifier = classifier)
