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

from util.helper_functions import Functions as f


class PoolLayer(object):
    """
    Layer that performs convolution and maxpooling/subsampling
    """

    def __init__(self, rng, input, subsample,filter_shape, image_shape, W = None, b = None,
            poolsize=(2, 2),maxoutsize = 1, params = {}, params_number = None):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        if W == None or b == None:
            W_name = "W" + str(params_number)
            b_name = "b" + str(params_number)

            if params.has_key(W_name) and params.has_key(b_name):
                # Initialize weights 
                W = theano.shared(
                    params[W_name],
                    name = W_name,
                    borrow=True
                )

                # Initialize biases
                b = theano.shared(params[b_name], name = b_name, borrow=True)
            else:
               # there are "num input feature maps * filter height * filter width"
               # inputs to each hidden unit
               fan_in = numpy.prod(filter_shape[1:])
               # each unit in the lower layer receives a gradient from:
               # "num output feature maps * filter height * filter width" /
               #   pooling size
               fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                          numpy.prod(poolsize))
               # initialize weights with random weights
               W_bound = numpy.sqrt(6. / (fan_in + fan_out))
               W = theano.shared(
                   numpy.asarray(
                       rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                       dtype=theano.config.floatX
                   ),
                   borrow=True
               )

               # the bias is a 1D tensor -- one bias per output feature map
               b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
               b = theano.shared(value=b_values, borrow=True)

        self.W = W
        self.b = b

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
        
        self.output = self.rectify(maxout_out)

        # Store parameters
        self.params = [self.W, self.b]
        
    # Rectify activation function
    def rectify(self,X): 
        return T.maximum(X,0.)

    def TestVersion(self,rng, input, subsample,filter_shape, image_shape, W = None, b = None,
            poolsize=(2, 2),maxoutsize = 1, params = {}, params_number = None):
        return PoolLayer(rng, input, subsample,filter_shape, image_shape, W = self.W, b = self.b,
            poolsize=poolsize, maxoutsize = maxoutsize, params = params, params_number = params_number)


