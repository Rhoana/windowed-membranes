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

    def __init__(self, input, n_in, n_out):

        # Initialize weights
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # Initialize biases
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        
        # Apply sigmoid function on output
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        # Define parameters in list
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        '''
        Return cost function
        '''
        # Calculate sum of derivatives
        test = self.p_y_given_x.reshape((self.p_y_given_x.shape[0],48,48))
        test_dx = (test[:,1:,:] - test[:,:-1,:]).reshape((self.p_y_given_x.shape[0],47*48))
        test_dy = (test[:,:,1:] - test[:,:,:-1]).reshape((self.p_y_given_x.shape[0],47*48))
        term = T.sum(T.abs_(test_dx)+T.abs_(test_dy),axis=1)
        
        # Calculate cost function
        factor = 1
        L = - T.sum( y* T.log(self.p_y_given_x) + (1 - y) * T.log(1 - self.p_y_given_x), axis=1)
        return T.mean(L+term*factor)

    def errors(self, y):
        '''
        Output errors
        '''
        prediction = T.round(self.p_y_given_x)
        L = T.sum(T.abs_(prediction-y),axis=1)
        return T.mean(L)/(48.*48.)
    
    def prediction(self):
        '''
        Output prediction
        '''
        return self.p_y_given_x
