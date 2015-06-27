import numpy as np

import util.helper_functions as f
import theano
from layers.pool_layer                    import PoolLayer
from layers.hidden_layer                  import HiddenLayer
from layers.logistic_sgd                  import LogisticRegression
from layers.pool_layer                    import PoolLayer
from layers.in_layer                      import InLayer

class FullyCon(object):
    '''
    Class that defines the hierarchy and design of the convolutional
    layers.
    '''
    
    def __init__(self, rng, batch_size,layers_3D,x,y,input_window_shape,output_window_shape,pred_window_size,classifier, params = None, test_version = False, network = None):
        
        if test_version == False:
            self.srng = theano.tensor.shared_randomstreams.RandomStreams(
                                rng.randint(999999))
                                
            in_layer = InLayer(batch_size,
                    input_window_shape,
                    output_window_shape,
                    pred_window_size,
                    layers_3D,
                    classifier)
                    
            in_layer.in_layer(x,y)
            y = in_layer.output_labeled

            self.layer0_input_size  = (batch_size,pred_window_size[0]**2)          
            
            # Initialize Layer 0
            #self.layer0_input = x.reshape(self.layer0_input_size)
            self.layer0_input = in_layer.output.reshape(self.layer0_input_size)

            self.layer0 = HiddenLayer(rng,
                                    input      = self.layer0_input,
                                    n_in       = pred_window_size[0]**2,
                                    n_out      = 1000,
                                    activation = f.rectify,
                                    params = params,
                                    params_number = 0)
                                    
            self.layer1 = HiddenLayer(rng,
                                    input      = f.dropout(self.layer0.output,p=0.5),
                                    n_in       = 1000,
                                    n_out      = 1000,
                                    activation = f.rectify,
                                    params = params,
                                    params_number = 1)
            
            self.layer2 = HiddenLayer(rng,
                                    input      = f.dropout(self.layer1.output,p=0.5),
                                    n_in       = 1000,
                                    n_out      = 1000,
                                    activation = f.rectify,
                                    params = params,
                                    params_number = 2)


            if classifier in ["membrane","synapse"]:
                self.layer4 = LogisticRegression(input = f.dropout(self.layer2.output,p=0.5),
                                            n_in  = 1000,
                                            n_out = pred_window_size[1]**2,
                                            y = y,
                                            out_window_shape = pred_window_size[1],
                                            params = params,
                                            params_number = 4,
                                            classifier = classifier)

            elif classifier == "membrane_synapse":
                self.layer4 = LogisticRegression(input = f.dropout(self.layer2.output,p=0.5),
                                            n_in  = 1000,
                                            n_out = 2*pred_window_size[1]**2,
                                            y = y,
                                            out_window_shape = pred_window_size[1],
                                            params = params,
                                            params_number = 4,
                                            classifier = classifier)
            
            # Define list of parameters
            self.params = self.layer0.params +self.layer1.params + self.layer2.params + self.layer4.params 

        else:
            self.srng = theano.tensor.shared_randomstreams.RandomStreams(
                                rng.randint(999999))

            self.layer0_input_size  = (batch_size,pred_window_size[0]**2)          
            
            # Initialize Layer 0
            self.layer0_input = x.reshape(self.layer0_input_size)


            self.layer0 = network.layer0.TestVersion(rng,
                                    input      = self.layer0_input,
                                    n_in       = pred_window_size[0]**2,
                                    n_out      = 1000,
                                    activation = f.rectify,
                                    params = params,
                                    params_number = 0)
                                    
            self.layer1 = network.layer1.TestVersion(rng,
                                    input      = self.layer0.output,
                                    n_in       = 1000,
                                    n_out      = 1000,
                                    activation = f.rectify,
                                    params = params,
                                    params_number = 1)
            
            self.layer2 = network.layer2.TestVersion(rng,
                                    input      = self.layer1.output,
                                    n_in       = 1000,
                                    n_out      = 1000,
                                    activation = f.rectify,
                                    params = params,
                                    params_number = 2)


            if classifier in ["membrane","synapse"]:
                self.layer4 = network.layer4.TestVersion(input = f.dropout(self.layer2.output,p=0.0),
                                            n_in  = 1000,
                                            n_out = pred_window_size[1]**2,
                                            y = y,
                                            out_window_shape = pred_window_size[1],
                                            params = params,
                                            params_number = 4,
                                            classifier = classifier)

            elif classifier == "membrane_synapse":
                self.layer4 = network.layer4.TestVersion(input = f.dropout(self.layer2.output,p=0.0),
                                            n_in  = 1000,
                                            n_out = 2*pred_window_size[1]**2,
                                            y = y,
                                            out_window_shape = pred_window_size[1],
                                            params = params,
                                            params_number = 4,
                                            classifier = classifier)
            
            # Define list of parameters
            self.params = self.layer0.params + self.layer4.params 
            
    def TestVersion(self,
            rng, 
            batch_size,
            layers_3D,
            x,
            y,
            input_window_shape,
            output_window_shape,
            pred_window_size,
            classifier,
            params = None, 
            network = None):

        return FullyCon(rng, 
                batch_size,
                layers_3D,
                x,
                y,
                input_window_shape,
                output_window_shape,
                pred_window_size,
                classifier,
                params = params, 
                test_version = True,
                network = network)
                
class FullyConCompressed(object):
    '''
    Class that defines the hierarchy and design of the convolutional
    layers.
    '''
    
    def __init__(self, 
        rng, 
        batch_size,
        layers_3D,
        x,
        y,
        input_window_shape,
        output_window_shape,
        pred_window_size,
        classifier, 
        params = None, 
        test_version = False, 
        network = None,
        compression = 4,
        version = 0):
        
        if version == 0:
            if test_version == False:
                self.srng = theano.tensor.shared_randomstreams.RandomStreams(
                                    rng.randint(999999))
                                
                in_layer = InLayer(batch_size,
                        input_window_shape,
                        output_window_shape,
                        pred_window_size,
                        layers_3D,
                        classifier)
                    
                in_layer.in_layer(x,y)
                y = in_layer.output_labeled
            
                #####################################################################
                # COMPRESS
                input_size  = (batch_size,pred_window_size[0],pred_window_size[0])  
                input = in_layer.output.reshape(input_size)
            
                compressed = input[:,:,::compression]
                dimension_k = int(np.floor(pred_window_size[0]/float(compression)))
                compressed_size = (batch_size,pred_window_size[0], dimension_k)
            
                self.layer0_input_size  = (batch_size,pred_window_size[0]*dimension_k)  
                self.layer0_input = compressed.reshape(self.layer0_input_size)
            
                ######################################################################

                self.layer0 = HiddenLayer(rng,
                                        input      = self.layer0_input,
                                        n_in       = pred_window_size[0]*dimension_k,
                                        n_out      = 4000,
                                        activation = f.rectify,
                                        params = params,
                                        params_number = 0)


                if classifier in ["membrane","synapse"]:
                    self.layer4 = LogisticRegression(input = f.dropout(self.layer0.output,p=0.5),
                                                n_in  = 4000,
                                                n_out = pred_window_size[1]**2,
                                                y = y,
                                                out_window_shape = pred_window_size[1],
                                                params = params,
                                                params_number = 4,
                                                classifier = classifier)

                elif classifier == "membrane_synapse":
                    self.layer4 = LogisticRegression(input = f.dropout(self.layer3.output,p=0.5),
                                                n_in  = 4000,
                                                n_out = 2*pred_window_size[1]**2,
                                                y = y,
                                                out_window_shape = pred_window_size[1],
                                                params = params,
                                                params_number = 4,
                                                classifier = classifier)
            
                # Define list of parameters
                self.params = self.layer0.params + self.layer4.params 

            else:
                self.srng = theano.tensor.shared_randomstreams.RandomStreams(
                                    rng.randint(999999))
                                
                                
                #####################################################################
                # COMPRESS
                input_size  = (batch_size,pred_window_size[0],pred_window_size[0])  
                input = x.reshape(input_size)
            
                compressed = input[:,:,::compression]
                dimension_k = int(np.floor(pred_window_size[0]/float(compression)))
                compressed_size = (batch_size,pred_window_size[0], dimension_k)
            
                self.layer0_input_size  = (batch_size,pred_window_size[0]*dimension_k)  
                self.layer0_input = compressed.reshape(self.layer0_input_size)
            
                #####################################################################$

                self.layer0 = network.layer0.TestVersion(rng,
                                        input      = self.layer0_input,
                                        n_in       = pred_window_size[0]*dimension_k,
                                        n_out      = 4000,
                                        activation = f.rectify,
                                        params = params,
                                        params_number = 0)


                if classifier in ["membrane","synapse"]:
                    self.layer4 = network.layer4.TestVersion(input = f.dropout(self.layer0.output,p=0.0),
                                                n_in  = 4000,
                                                n_out = pred_window_size[1]**2,
                                                y = y,
                                                out_window_shape = pred_window_size[1],
                                                params = params,
                                                params_number = 4,
                                                classifier = classifier)

                elif classifier == "membrane_synapse":
                    self.layer4 = network.layer4.TestVersion(input = f.dropout(self.layer3.output,p=0.0),
                                                n_in  = 4000,
                                                n_out = 2*pred_window_size[1]**2,
                                                y = y,
                                                out_window_shape = pred_window_size[1],
                                                params = params,
                                                params_number = 4,
                                                classifier = classifier)
            
                # Define list of parameters
                self.params = self.layer0.params + self.layer4.params 
                
        if version == 1:
            if test_version == False:
                self.srng = theano.tensor.shared_randomstreams.RandomStreams(
                                    rng.randint(999999))
                                
                in_layer = InLayer(batch_size,
                        input_window_shape,
                        output_window_shape,
                        pred_window_size,
                        layers_3D,
                        classifier)
                    
                in_layer.in_layer(x,y)
                y = in_layer.output_labeled
            
                #####################################################################
                # COMPRESS
                input_size  = (batch_size,pred_window_size[0],pred_window_size[0])  
                input = in_layer.output.reshape(input_size)
            
                compressed = input[:,:,::compression]
                dimension_k = int(np.floor(pred_window_size[0]/float(compression)))
                compressed_size = (batch_size,pred_window_size[0], dimension_k)
            
                self.layer0_input_size  = (batch_size,pred_window_size[0]*dimension_k)  
                self.layer0_input = compressed.reshape(self.layer0_input_size)
            
                ######################################################################
                
                self.layer1 = HiddenLayer(rng,
                                        input      = self.layer0_input,
                                        n_in       = pred_window_size[0]*dimension_k,
                                        n_out      = pred_window_size[0]**2,
                                        activation = f.rectify,
                                        params = params,
                                        params_number = 0)

                self.layer1 = HiddenLayer(rng,
                                        input      = self.layer1.output,
                                        n_in       = pred_window_size[0]*pred_window_size[0],
                                        n_out      = 4000,
                                        activation = f.rectify,
                                        params = params,
                                        params_number = 1)


                if classifier in ["membrane","synapse"]:
                    self.layer4 = LogisticRegression(input = f.dropout(self.layer1.output,p=0.5),
                                                n_in  = 4000,
                                                n_out = pred_window_size[1]**2,
                                                y = y,
                                                out_window_shape = pred_window_size[1],
                                                params = params,
                                                params_number = 4,
                                                classifier = classifier)

                elif classifier == "membrane_synapse":
                    self.layer4 = LogisticRegression(input = f.dropout(self.layer1.output,p=0.5),
                                                n_in  = 4000,
                                                n_out = 2*pred_window_size[1]**2,
                                                y = y,
                                                out_window_shape = pred_window_size[1],
                                                params = params,
                                                params_number = 4,
                                                classifier = classifier)
            
                # Define list of parameters
                self.params = self.layer0.params + self.layer1.params + self.layer4.params 

            else:
                self.srng = theano.tensor.shared_randomstreams.RandomStreams(
                                    rng.randint(999999))
                                
                                
                #####################################################################
                # COMPRESS
                input_size  = (batch_size,pred_window_size[0],pred_window_size[0])  
                input = x.reshape(input_size)
            
                compressed = input[:,:,::compression]
                dimension_k = int(np.floor(pred_window_size[0]/float(compression)))
                compressed_size = (batch_size,pred_window_size[0], dimension_k)
            
                self.layer0_input_size  = (batch_size,pred_window_size[0]*dimension_k)  
                self.layer0_input = compressed.reshape(self.layer0_input_size)
            
                #####################################################################$

                self.layer0 = network.layer0.TestVersion(rng,
                                        input      = self.layer0_input,
                                        n_in       = pred_window_size[0]*dimension_k,
                                        n_out      = pred_window_size[0]*pred_window_size[0],
                                        activation = f.rectify,
                                        params = params,
                                        params_number = 0)
                                        
                self.layer1 = network.layer1.TestVersion(rng,
                                        input      = self.layer0.output,
                                        n_in       = pred_window_size[0]**2,
                                        n_out      = 4000,
                                        activation = f.rectify,
                                        params = params,
                                        params_number = 0)


                if classifier in ["membrane","synapse"]:
                    self.layer4 = network.layer4.TestVersion(input = f.dropout(self.layer1.output,p=0.0),
                                                n_in  = 4000,
                                                n_out = pred_window_size[1]**2,
                                                y = y,
                                                out_window_shape = pred_window_size[1],
                                                params = params,
                                                params_number = 4,
                                                classifier = classifier)

                elif classifier == "membrane_synapse":
                    self.layer4 = network.layer4.TestVersion(input = f.dropout(self.layer1.output,p=0.0),
                                                n_in  = 4000,
                                                n_out = 2*pred_window_size[1]**2,
                                                y = y,
                                                out_window_shape = pred_window_size[1],
                                                params = params,
                                                params_number = 4,
                                                classifier = classifier)
            
                # Define list of parameters
                self.params = self.layer0.params + self.layer1.params + self.layer4.params
            
    def TestVersion(self,
            rng, 
            batch_size,
            layers_3D,
            x,
            y,
            input_window_shape,
            output_window_shape,
            pred_window_size,
            classifier,
            params = None, 
            network = None):

        return FullyConCompressed(rng, 
                batch_size,
                layers_3D,
                x,
                y,
                input_window_shape,
                output_window_shape,
                pred_window_size,
                classifier,
                params = params, 
                test_version = True,
                network = network)
 
