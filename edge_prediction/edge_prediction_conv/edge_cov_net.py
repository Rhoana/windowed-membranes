from helper_functions import Functions
import theano

from lib.pool_layer import PoolLayer

class Convolution(Functions):
    '''
    Class that defines the hierarchy and design of the convolutional
    layers.
    '''
    
    def __init__(self, rng, batch_size,num_kernels,kernel_sizes,x,y):
        
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(
                            rng.randint(999999))
        self.layer0_input_size  = (batch_size, 1, 48, 48)                             # Input size from data 
        self.edge0              = (48 - kernel_sizes[0][0] + 1)/ 2                    # New edge size
        self.layer0_output_size = (batch_size, num_kernels[0], self.edge0, self.edge0)  # Output size
        assert ((48 - kernel_sizes[0][0] + 1) % 2) == 0                                # Check pooling size
        
        # Initialize Layer 0
        #self.layer0_input = x.reshape(self.layer0_input_size)
        self.layer0_input = x.reshape((batch_size,1,48,48))
        self.layer0 = PoolLayer(rng,
                                    input=self.dropout(self.layer0_input,p=0.2),
                                    image_shape=self.layer0_input_size,
                                    subsample= (1,1),
                                    filter_shape=(num_kernels[0], 1) + kernel_sizes[0],
                                    poolsize=(2, 2))

        self.layer1_input_size  = self.layer0_output_size                              # Input size Layer 1
        self.edge1              = (self.edge0 - kernel_sizes[1][0] + 1)/ 2            # New edge size
        self.layer1_output_size = (batch_size, num_kernels[1], self.edge1, self.edge1) # Output size
        assert ((self.edge0 - kernel_sizes[1][0] + 1) % 2) == 0                        # Check pooling size

        # Initialize Layer 1
        self.layer1 = PoolLayer(rng,
                                    input= self.dropout(self.layer0.output,p=0.2),
                                    image_shape=self.layer1_input_size,
                                    subsample= (1,1),
                                    filter_shape=(num_kernels[1], num_kernels[0]) + kernel_sizes[1],
                                    poolsize=(2, 2))
                                    
        self.layer2_input_size  = self.layer1_output_size                              # Input size Layer 1
        self.edge2              = (self.edge1 - kernel_sizes[2][0] + 1)/2           # New edge size
        self.layer2_output_size = (batch_size, num_kernels[2], self.edge2, self.edge2) # Output size
        assert ((self.edge1 - kernel_sizes[2][0] + 1) % 2) == 0                        # Check pooling size

        # Initialize Layer 1
        self.layer2 = PoolLayer(rng,
                                    input= self.dropout(self.layer1.output,p=0.2),
                                    image_shape=self.layer2_input_size,
                                    subsample= (1,1),
                                    filter_shape=(num_kernels[2], num_kernels[1]) + kernel_sizes[2],
                                    poolsize=(2, 2))

