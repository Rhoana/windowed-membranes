import os
import sys
import time
import numpy
import theano
import theano.tensor as T

from lib import init

class LogisticRegression(object):
    """
    Logistic regression class
    """

    def __init__(self, input, n_in, n_out,y, out_window_shape, classes = 2, W = None, b = None,
            params = {}, params_number = None, classifier = 'standard'):

        self.out_window_shape = out_window_shape
        self.classifier = classifier
        self.y = y

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
                
                #W = theano.shared(init.HeNormal((n_in, n_out)), borrow=True, name = W_name)
                #b = theano.shared(init.constant((n_out,), 0.), borrow=True, name = b_name)

        self.W = W
        self.b = b
        
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        # Define parameters in list
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, penatly_factor):
        '''
        Return cost function
        '''
        # Calculate sum of derivatives
        #if self.classifier in ["membrane","synapse"]:
        #    test = self.p_y_given_x.reshape((self.p_y_given_x.shape[0],1,self.out_window_shape,self.out_window_shape))
        #    test_dx = (test[:,:,1:,:] - test[:,:,:-1,:]).reshape((self.p_y_given_x.shape[0],(self.out_window_shape-1)*self.out_window_shape))
        #    test_dy = (test[:,:,:,1:] - test[:,:,:,:-1]).reshape((self.p_y_given_x.shape[0],(self.out_window_shape-1)*self.out_window_shape))
        #elif self.classifier == "membrane_synapse":
        #    test = self.p_y_given_x.reshape((self.p_y_given_x.shape[0],2,self.out_window_shape,self.out_window_shape))
        #    test_dx = (test[:,:,1:,:] - test[:,:,:-1,:]).reshape((self.p_y_given_x.shape[0],2*(self.out_window_shape-1)*self.out_window_shape))
        #    test_dy = (test[:,:,:,1:] - test[:,:,:,:-1]).reshape((self.p_y_given_x.shape[0],2*(self.out_window_shape-1)*self.out_window_shape))

        #term = T.mean(T.abs_(test_dx)+T.abs_(test_dy),axis=1)
        
        # Calculate cost function
        L = - T.mean( self.y* T.log(self.p_y_given_x) + (1 - self.y) * T.log(1 - self.p_y_given_x), axis=1)
        return T.mean(L) #+term*penatly_factor)

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

    def TestVersion(self,
            input, 
            n_in, 
            n_out, 
            y,
            out_window_shape, 
            classes = 2, 
            W = None, 
            b = None,
            params = {}, 
            params_number = None, 
            classifier = 'standard'):
        return LogisticRegression(input, 
                n_in, 
                n_out, 
                y,
                out_window_shape, 
                classes = classes, 
                W = self.W, 
                b = self.b,
                params = params, 
                params_number = params_number, 
                classifier = classifier)




