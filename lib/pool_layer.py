import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from hidden_layer import HiddenLayer

from edge_prediction_conv.helper_functions import Functions as f


class PoolLayer(object):
    """
    Layer that performs convolution and maxpooling/subsampling
    """

    def __init__(self, rng, input, subsample,filter_shape, image_shape, poolsize=(2, 2),maxoutsize = 1):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # Design random matrix
        random_matrix = f.make_random_matrix(filter_shape, poolsize)

        # Initialize weights 
        self.W = theano.shared(
            numpy.asarray(random_matrix,dtype=theano.config.floatX),
            borrow=True
        )

        # Initialize biases
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # Convolutional filter
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            subsample=subsample,
            image_shape=image_shape
        )
        

        # Downsampling using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # Rectify activation function
        def rectify(X): 
            return T.maximum(X,0.)
        
        # Define output 
        #self.output = rectify(pooled_out + self.b.dimshuffle('x', 0,'x','x'))
        bias_out = pooled_out + self.b.dimshuffle('x', 0,'x','x')
        
        # Maxout
        maxout_out = None
        for i in xrange(maxoutsize):
            t = bias_out[:,i::maxoutsize,:,:]
            if maxout_out is None:
                maxout_out = t
            else:
                maxout_out = T.maximum(maxout_out, t)
        
        self.output = rectify(maxout_out)

        # Store parameters
        self.params = [self.W, self.b]
        
