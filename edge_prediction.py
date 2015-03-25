import sys
import time as time
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T


matplotlib.pyplot.gray()
theano.config.floatX = 'float32'
rng = np.random.RandomState(42)

from convolutional_mlp  import LeNetConvPoolLayer
from mlp                import HiddenLayer
from logistic_sgd       import LogisticRegression
from pre_process        import PreProcess
from synapse_train_data.read_img import * 

class Functions(object):
    '''
    Class containing helper functions for the ConvNet class.
    '''
    
    def dropout(self,X,p=0.5):
        '''
        Perform dropout with probability p
        '''
        if p>0:
            retain_prob = 1-p
            X *= self.srng.binomial(X.shape,p=retain_prob,dtype = theano.config.floatX)
            X /= retain_prob
        return X
        
    def vstack(self,layers):
        '''
        Vstack
        '''
        n = 0
        for layer in layers:
            if n == 1:
                out_layer = T.concatenate(layer,layers[n-1])
            elif n>1:
                out_layer = T.concatenate(out_layer,layer)
            n += 1
        return out_layer

    def rectify(self,X): 
        '''
        Rectified linear activation function
        '''
        return T.maximum(X,0.)
        
    def RMSprop(self,cost, params, lr = 0.001, rho=0.9, epsilon=1e-6):
        '''
        RMSprop - optimization (http://nbviewer.ipython.org/github/udibr/Theano-Tutorials/blob/master/notebooks/4_modern_net.ipynb)
        '''
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc              = theano.shared(p.get_value() * 0.)
            acc_new          = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g                = g / gradient_scaling
            
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
            
        return updates
    
    def stochasticGradient(self,cost,params,lr):
        '''
        Stochastic Gradient Descent
        '''
        updates = [
            (param_i, param_i - lr * grad_i)  # <=== SGD update step
            for param_i, grad_i in zip(params, grads)
        ]
        return updates       
        
    def init_optimizer(self, optimizer, cost, params, optimizerData):
        '''
        Choose between different optimizers 
        '''
        if optimizer == 'stochasticGradient':
            updates = self.stochasticGradient(cost, 
                                              params,
                                              lr      = optimizerData['learning_rate'])
        elif optimizer == 'RMSprop':    
            updates = self.RMSprop(cost, params, optimizerData['learning_rate'],
                                                 rho     = optimizerData['rho'],
                                                 epsilon = optimizerData['epsilon'])
                                                 
        return updates
        
        
class Convolution(Functions):
    '''
    Class that defines the hierarchy and design of the convolutional
    layers.
    '''
    
    def __init__(self,batch_size,num_kernels,kernel_sizes,x,y):
        
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(
                            rng.randint(999999))
        self.layer0_input_size  = (batch_size, 1, 48, 48)                             # Input size from data 
        self.edge0              = (48 - kernel_sizes[0][0] + 1)/ 2                    # New edge size
        self.layer0_output_size = (batch_size, num_kernels[0], self.edge0, self.edge0)  # Output size
        assert ((48 - kernel_sizes[0][0] + 1) % 2) == 0                                # Check pooling size
        
        # Initialize Layer 0
        #self.layer0_input = x.reshape(self.layer0_input_size)
        self.layer0_input = x.reshape((batch_size,1,48,48))
        self.layer0 = LeNetConvPoolLayer(rng,
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
        self.layer1 = LeNetConvPoolLayer(rng,
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
        self.layer2 = LeNetConvPoolLayer(rng,
                                    input= self.dropout(self.layer1.output,p=0.2),
                                    image_shape=self.layer2_input_size,
                                    subsample= (1,1),
                                    filter_shape=(num_kernels[2], num_kernels[1]) + kernel_sizes[2],
                                    poolsize=(2, 2))


class ConvNet(Functions):
    '''
    Main function for the convolutional network
    '''
    
    def __init__(self):
        
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(
                            rng.randint(999999))
                            
    
    def stack(self,stackitems):
        '''
        Works like vstack for theano tensors 
        '''
        for n in xrange(len(stackitems)-1):
            if n == 0:
                output = T.concatenate((stackitems[n],stackitems[n+1]),axis=1)
            else:
                output = T.concatenate((output,stackitems[n+1]),axis=1)
        return output
                            
    
    def model(self,batch_size,num_kernels,kernel_sizes,x,y):
        '''
        Defining convolutional architecture
        '''
        
        # Convolutional layers
        conv  = Convolution(batch_size, num_kernels, kernel_sizes, x, y)
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
            num_kernels  = [10,10,10],
            kernel_sizes = [(5, 5), (3, 3), (3,3)],
            batch_size   = 50,
            epochs       = 100,
            optimizer    = 'RMSprop'):
            
            
        optimizerData = {}
        optimizerData['learning_rate'] = 0.001
        optimizerData['rho']           = 0.9
        optimizerData['epsilon']       = 1e-4
        
        if len(sys.argv) > 1 and sys.argv[1] == "--pre-process":
            print "Generating Train/Test Set..."
            generate_training_set()

        print 'Loading data ...'
        
        # load in and process data
        preProcess              = PreProcess()
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
        
        output = output.reshape(output.shape[0],48,48)
        
        np.save('output.npy',output)
        np.save('x.py',test_set_x.eval())
        np.save('y.npy',test_set_y.eval())
        
        import matplotlib.pyplot as plt

        fig = plt.figure(2)
        ax1 = fig.add_subplot(132)
        ax1.imshow(test_set_y.eval()[0].reshape(48,48),cmap=plt.cm.gray)
        
        ax2 = fig.add_subplot(133)
        ax2.imshow(output[0],cmap=plt.cm.gray)
        
        ax3 = fig.add_subplot(131)
        ax3.imshow(test_set_x.eval()[0].reshape(48,48),cmap=plt.cm.gray)
        
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)
        
        
        ax3.set_title('Input Image')
        ax2.set_title('Labeled Edges')
        ax1.set_title('Predicted Edges')
        
        plt.show()
        

if __name__ == "__main__":
    convnet = ConvNet()
    convnet.run()
    
    
    
    
    
    

