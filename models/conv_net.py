from util.helper_functions import Functions
import theano
from layers.pool_layer                    import PoolLayer
from layers.hidden_layer                  import HiddenLayer
from layers.logistic_sgd                  import LogisticRegression
from layers.pool_layer                    import PoolLayer
from layers.in_layer                      import InLayer

class ConvNet(Functions):
    '''
    Class that defines the hierarchy and design of the convolutional
    layers.
    '''
    
    def __init__(self, rng, batch_size,layers_3D,num_kernels,kernel_sizes,x,y,input_window_shape,output_window_shape,pred_window_size,classifier,maxoutsize = (1,1,1), params = None, dropout = [0.0,0.0,0.0,0.0], test_version = False, network = None):
        
        if test_version == False:
            self.srng = theano.tensor.shared_randomstreams.RandomStreams(
                                rng.randint(999999))
                                
            in_layer = InLayer(batch_size,
                    input_window_shape,
                    output_window_shape,
                    pred_window_size,
                    layers_3D)
                    
            in_layer.in_layer(x,y)
            y = in_layer.output_labeled

            self.layer0_input_size  = (batch_size,layers_3D,pred_window_size[0],pred_window_size[0])          
            self.edge0              = (pred_window_size[0] - kernel_sizes[0][0] + 1)/ 2                     # New edge size
            self.layer0_output_size = (batch_size, num_kernels[0]/maxoutsize[0], self.edge0, self.edge0)                    # Output size
            assert ((pred_window_size[0] - kernel_sizes[0][0] + 1) % 2) == 0                                # Check pooling size
            
            # Initialize Layer 0
            #self.layer0_input = x.reshape(self.layer0_input_size)
            self.layer0_input = in_layer.output

            self.layer0 = PoolLayer(rng,
                                        input=self.layer0_input,
                                        image_shape=self.layer0_input_size,
                                        subsample= (1,1),
                                        filter_shape=(num_kernels[0], layers_3D) + kernel_sizes[0],
                                        poolsize=(2, 2),
                                        maxoutsize = maxoutsize[0],
                                        params = params,
                                        params_number = 0)

            self.layer1_input_size  = self.layer0_output_size                              # Input size Layer 1
            self.edge1              = (self.edge0 - kernel_sizes[1][0] + 1)/ 2            # New edge size
            self.layer1_output_size = (batch_size, num_kernels[1]/maxoutsize[1], self.edge1, self.edge1) # Output size
            assert ((self.edge0 - kernel_sizes[1][0] + 1) % 2) == 0                        # Check pooling size

            # Initialize Layer 1
            self.layer1 = PoolLayer(rng,
                                        input= self.dropout(self.layer0.output,p=dropout[0]),
                                        image_shape=self.layer1_input_size,
                                        subsample= (1,1),
                                        filter_shape=(num_kernels[1], num_kernels[0]/maxoutsize[0]) + kernel_sizes[1],
                                        poolsize=(2, 2),
                                        maxoutsize = maxoutsize[1],
                                        params = params,
                                        params_number = 1)
                                        
            self.layer2_input_size  = self.layer1_output_size                              # Input size Layer 1
            self.edge2              = (self.edge1 - kernel_sizes[2][0] + 1)          # New edge size
            self.layer2_output_size = (batch_size, num_kernels[2]/maxoutsize[2], self.edge2, self.edge2) # Output size
            #assert (self.edge1 - kernel_sizes[2][0] + 1) == 0                        # Check pooling size

            # Initialize Layer 2
            self.layer2 = PoolLayer(rng,
                                        input= self.dropout(self.layer1.output,p=dropout[1]),
                                        image_shape=self.layer2_input_size,
                                        subsample= (1,1),
                                        filter_shape=(num_kernels[2], num_kernels[1]/maxoutsize[1]) + kernel_sizes[2],
                                        poolsize=(1, 1),
                                        maxoutsize = maxoutsize[2],
                                        params = params,
                                        params_number = 2)

            self.layer3_input = self.layer2.output.flatten(2)
            
            # Layer 3: Fully connected layer
            self.layer3 = HiddenLayer(rng,
                                    input      = self.dropout(self.layer3_input,p=dropout[2]),
                                    n_in       = (num_kernels[2]/maxoutsize[2]) * self.edge2 * self.edge2,
                                    n_out      = (num_kernels[2]/maxoutsize[2]) * self.edge2 * self.edge2,
                                    activation = self.rectify,
                                    params = params,
                                    params_number = 3)


            self.layer4 = LogisticRegression(input = self.dropout(self.layer3.output,p=dropout[3]),
                                            n_in  = (num_kernels[2]/maxoutsize[2]) * self.edge2 * self.edge2,
                                            n_out = pred_window_size[1]**2,
                                            y = y,
                                            out_window_shape = pred_window_size[1],
                                            params = params,
                                            params_number = 4,
                                            classifier = classifier)
            
            # Define list of parameters
            convparams = self.layer2.params + self.layer1.params + self.layer0.params
            hiddenparams = self.layer4.params + self.layer3.params
            self.params = convparams + hiddenparams 

        else:
            self.srng = theano.tensor.shared_randomstreams.RandomStreams(
                                rng.randint(999999))

            self.layer0_input_size  = (batch_size,layers_3D,pred_window_size[0],pred_window_size[0])   
            self.edge0              = (pred_window_size[0] - kernel_sizes[0][0] + 1)/ 2                     # New edge size
            self.layer0_output_size = (batch_size, num_kernels[0]/maxoutsize[0], self.edge0, self.edge0)                    # Output size
            assert ((pred_window_size[0] - kernel_sizes[0][0] + 1) % 2) == 0                                # Check pooling size
            
            # Initialize Layer 0
            self.layer0_input = x.reshape(self.layer0_input_size)

            self.layer0 = network.layer0.TestVersion(rng,
                                        input=self.layer0_input,
                                        image_shape=self.layer0_input_size,
                                        subsample= (1,1),
                                        filter_shape=(num_kernels[0], layers_3D) + kernel_sizes[0],
                                        poolsize=(2, 2),
                                        maxoutsize = maxoutsize[0],
                                        params = params,
                                        params_number = 0)

            self.layer1_input_size  = self.layer0_output_size                              # Input size Layer 1
            self.edge1              = (self.edge0 - kernel_sizes[1][0] + 1)/ 2            # New edge size
            self.layer1_output_size = (batch_size, num_kernels[1]/maxoutsize[1], self.edge1, self.edge1) # Output size
            assert ((self.edge0 - kernel_sizes[1][0] + 1) % 2) == 0                        # Check pooling size

            # Initialize Layer 1
            self.layer1 = network.layer1.TestVersion(rng,
                                        input= self.dropout(self.layer0.output,p=dropout[0]),
                                        image_shape=self.layer1_input_size,
                                        subsample= (1,1),
                                        filter_shape=(num_kernels[1], num_kernels[0]/maxoutsize[0]) + kernel_sizes[1],
                                        poolsize=(2, 2),
                                        maxoutsize = maxoutsize[1],
                                        params = params,
                                        params_number = 1)
                                        
            self.layer2_input_size  = self.layer1_output_size                              # Input size Layer 1
            self.edge2              = (self.edge1 - kernel_sizes[2][0] + 1)          # New edge size
            self.layer2_output_size = (batch_size, num_kernels[2]/maxoutsize[2], self.edge2, self.edge2) # Output size
            #assert (self.edge1 - kernel_sizes[2][0] + 1) == 0                        # Check pooling size

            # Initialize Layer 2
            self.layer2 = network.layer2.TestVersion(rng,
                                        input= self.dropout(self.layer1.output,p=dropout[1]),
                                        image_shape=self.layer2_input_size,
                                        subsample= (1,1),
                                        filter_shape=(num_kernels[2], num_kernels[1]/maxoutsize[1]) + kernel_sizes[2],
                                        poolsize=(1, 1),
                                        maxoutsize = maxoutsize[2],
                                        params = params,
                                        params_number = 2)

            self.layer3_input = self.layer2.output.flatten(2)
            
            # Layer 3: Fully connected layer
            self.layer3 = network.layer3.TestVersion(rng,
                                    input      = self.dropout(self.layer3_input,p=dropout[2]),
                                    n_in       = (num_kernels[2]/maxoutsize[2]) * self.edge2 * self.edge2,
                                    n_out      = (num_kernels[2]/maxoutsize[2]) * self.edge2 * self.edge2,
                                    activation = self.rectify,
                                    params = params,
                                    params_number = 3)


            self.layer4 = network.layer4.TestVersion(input = self.dropout(self.layer3.output,p=dropout[3]),
                                            n_in  = (num_kernels[2]/maxoutsize[2]) * self.edge2 * self.edge2,
                                            n_out = pred_window_size[1]**2,
                                            y = y,
                                            out_window_shape = pred_window_size[1],
                                            params = params,
                                            params_number = 4,
                                            classifier = classifier)
            
    def TestVersion(self,
            rng, 
            batch_size,
            layers_3D,
            num_kernels,
            kernel_sizes,
            x,
            y,
            input_window_shape,
            output_window_shape,
            pred_window_size,
            classifier,
            maxoutsize = (1,1,1), 
            params = None, 
            dropout = [0.0,0.0,0.0,0.0], 
            network = None):

        return ConvNet(rng, 
                batch_size,
                layers_3D,
                num_kernels,
                kernel_sizes,
                x,
                y,
                input_window_shape,
                output_window_shape,
                pred_window_size,
                classifier,
                maxoutsize = maxoutsize, 
                params = params, 
                dropout = dropout, 
                test_version = True,
                network = network)
 
 


