import os
import sys
import time

import numpy

import theano
import theano.tensor as T


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Fully connected layer 
        """
        
        # Define input
        self.input = input
       
        # Initialize weights
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low  = -numpy.sqrt(6. / (n_in + n_out)),
                    high = numpy.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name='W', borrow=True)
        
        # Initialize biases
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        
        # Calculate output 
        lin_output = T.dot(input, self.W) + self.b
        self.output = activation(lin_output)

        # Add parameters to model
        self.params = [self.W, self.b]

