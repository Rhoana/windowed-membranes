import os
import sys
import time

import numpy

import theano
import theano.tensor as T


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
            activation=T.tanh, params = {}, params_number = None):
        """
        Fully connected layer 
        """
        
        # Define input
        self.input = input

        W_name = "W" + str(params_number)
        b_name = "b" + str(params_number)
       
        # Initialize weights
        if params.has_key(W_name) and params.has_key(b_name): 
            W = theano.shared(params[W_name], name=W_name, borrow=True)
        
            # Initialize biasea
            b = theano.shared(params[b_name], name=b_name, borrow=True)

        else:
            W_values = numpy.asarray(
                rng.uniform(
                    low  = -numpy.sqrt(6. / (n_in + n_out)),
                    high = numpy.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name=W_name, borrow=True)
        
            # Initialize biasea
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=b_name, borrow=True)


        self.W = W
        self.b = b
        
        # Calculate output 
        lin_output = T.dot(input, self.W) + self.b
        self.output = activation(lin_output)

        # Add parameters to model
        self.params = [self.W, self.b]

