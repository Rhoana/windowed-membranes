import sys
import time as time
import os

# PLT doesnt play nice with Tesla K40m
#import matplotlib
#import matplotlib.pyplot as plt 
import numpy as np
import theano
import theano.tensor as T

# matplotlib.pyplot.gray()

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

### IMPORTING FROM A DIRECTORY ONE LEVEL UP!
from lib.pool_layer import PoolLayer
from lib.hidden_layer import HiddenLayer
from lib.logistic_sgd       import LogisticRegression

from edge_prediction.util.pre_process        import PreProcess
from edge_prediction.data.read_img import * 
from edge_prediction.edge_prediction_conv.helper_functions import Functions
from edge_prediction.edge_prediction_conv.edge_cov_net import Convolution

class ConvNet(Functions):
    '''
    Main function for the convolutional network
    '''
    
    def __init__(self):
        
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(
                            rng.randint(999999))
                            
                            
    
    def model(self,batch_size,num_kernels,kernel_sizes,x,y):
        '''
        Defining convolutional architecture
        '''
        
        # Convolutional layers
        conv  = Convolution(rng, batch_size, num_kernels, kernel_sizes, x, y)
        layer3_input = conv.layer2.output.flatten(2)
        
        # Fully connected layer
        layer3 = HiddenLayer(rng,
                                  input      = self.dropout(layer3_input,p=0.2),
                                  n_in       = num_kernels[2] * conv.edge2 * conv.edge2,
                                  n_out      = num_kernels[2] * conv.edge2 * conv.edge2,
                                  activation = self.rectify)


        # Logistic regression layer
        layer4 = LogisticRegression(input = self.dropout(layer3.output,p=0.2),
                                         n_in  = num_kernels[2] * conv.edge2 * conv.edge2,
                                         n_out = 48*48)
        
        # Define list of parameters
        convparams = conv.layer2.params +conv.layer1.params + conv.layer0.params
        hiddenparams = layer4.params + layer3.params
        self.params = hiddenparams + convparams 
        self.conv   = conv
        self.layer3 = layer3
        self.layer4 = layer4
                                         

    def run(self,
            net_size = 'small',
            batch_size   = 50,
            epochs       = 100,
            optimizer    = 'RMSprop'):
        

        optimizerData = {}
        optimizerData['learning_rate'] = 0.001
        optimizerData['rho']           = 0.9
        optimizerData['epsilon']       = 1e-4
        
        if len(sys.argv) > 1 and ( "--pre-process" in sys.argv):
            print "Generating Train/Test Set..."
            generate_training_set()

        if len(sys.argv) > 1 and ( "--small" in sys.argv):
            num_kernels   = [10,10,10]
            kernel_sizes  = [(5, 5), (3, 3), (3,3)]
            train_samples = 1500
            val_samples   = 200
            test_samples  = 500
        elif len(sys.argv) > 1 and ( "--medium" in sys.argv):
            num_kernels  = [64,64,64]
            kernel_sizes = [(5, 5), (3, 3), (3,3)]
            train_samples = 4000
            val_samples   = 200
            test_samples  = 1000
        elif len(sys.argv) > 1 and ( "--large" in sys.argv):
            num_kernels  = [80,80,80]
            kernel_sizes = [(5, 5), (3, 3), (3,3)]
            train_samples = 9000
            val_samples   = 200
            test_samples  = 1000
        else:
            print 'Error: pass network size (small/medium/large)'
            exit()

        print 'Loading data ...'
        
        # load in and process data
        preProcess              = PreProcess(train_samples,val_samples,test_samples)
        data                    = preProcess.run()
        train_set_x,train_set_y = data[0],data[3]
        valid_set_x,valid_set_y = data[1],data[4]
        test_set_x,test_set_y   = data[2],data[5]

        print 'Initializing neural network ...'
    
        # print error if batch size is to large
        if valid_set_y.eval().size<batch_size:
            print 'Error: Batch size is larger than size of validation set.'

        # compute batch sizes for train/test/validation
        n_train_batches  = train_set_x.get_value(borrow=True).shape[0]
        n_test_batches   = test_set_x.get_value(borrow=True).shape[0]
        n_valid_batches  = valid_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_test_batches  /= batch_size
        n_valid_batches /= batch_size

        # symbolic variables
        x = T.matrix('x')  # input image data
        y = T.matrix('y')  # input label data
        
        self.model(batch_size, num_kernels, kernel_sizes, x, y)

        # Initialize parameters and functions
        cost   = self.layer4.negative_log_likelihood(y)        # Cost function
        params = self.params                                   # List of parameters
        grads  = T.grad(cost, params)                          # Gradient
        index  = T.lscalar()                                   # Index
        
        # Intialize optimizer
        updates = self.init_optimizer(optimizer, cost, params, optimizerData)

        # Training model
        train_model = theano.function(
                      [index],
                      cost,
                      updates = updates,
                      givens  = {
                                x: train_set_x[index * batch_size: (index + 1) * batch_size], 
                                y: train_set_y[index * batch_size: (index + 1) * batch_size] 
            }
        )

        #Validation function
        validate_model = theano.function(
                         [index],
                         self.layer4.errors(y),
                         givens = {
                                  x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                  y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )

        #Test function
        test_model = theano.function(
                     [index],
                     self.layer4.errors(y),
                     givens = {
                              x: test_set_x[index * batch_size: (index + 1) * batch_size],
                              y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        
        # Solver
        try:
            print '... Solving'
            start_time = time.time()    
            for epoch in range(epochs):
                t1 = time.time()
                costs             = [train_model(i) for i in xrange(n_train_batches)]
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                t2 = time.time()
                print "Epoch {}    NLL {:.2}    %err in validation set {:.1%}    Time (epoch/total) {:.2}/{:.2} mins".format(epoch + 1, np.mean(costs), np.mean(validation_losses),(t2-t1)/60.,(t2-start_time)/60.)
        except KeyboardInterrupt:
            print 'Exiting solver ...'
        #Evaluate performance 
        test_errors = [test_model(i) for i in range(n_test_batches)]
        print "test errors: {:.1%}".format(np.mean(test_errors))
        
        predict = theano.function(inputs=[index], 
                                    outputs=self.layer4.prediction(),
                                    givens = {
                                        x: test_set_x[index * batch_size: (index + 1) * batch_size]
                                        }
                                    )
                                    
        # Plot example of output
        output = np.zeros((0,48*48))
        for i in xrange(n_test_batches):
            output = np.vstack((output,predict(i)))
        
        shape = (output.shape[0],48,48)
        output = output.reshape(shape)

        if not os.path.exists('results'):
            os.makedirs('results')
        
        np.save('results/output.npy',output)
        np.save('results/x.npy',test_set_x.eval().reshape(shape))
        np.save('results/y.npy',test_set_y.eval().reshape(shape))

        # from plot import plot
        # plot()
        # plt.show()

if __name__ == "__main__":
    # GLOBALS
    theano.config.floatX = 'float32'
    rng = np.random.RandomState(42)

    convnet = ConvNet()
    convnet.run()
    
    
    
    
    
    

