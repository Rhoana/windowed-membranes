import sys
import time as time
import os
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from util.pre_process                  import PreProcess 
from data.read_img                     import Read 
from edge_prediction_conv.edge_cov_net import CovNet 
from edge_prediction_conv.helper_functions import Functions as f 
import util.post_process as post 
import cPickle

class ConvNetClassifier(object):

    def __init__(self,params = {}):
        self.params = params

    def process_cmd_line_args(self,in_window_shape,out_window_shape,stride,img_size, classifier, n_train_files, n_test_files, samples_per_image, on_ratio, membrane_edges):
        
        if len(sys.argv) > 1 and ( "--small" in sys.argv):
            num_kernels   = [10,10,10]
            kernel_sizes  = [(5, 5), (3, 3), (3,3)]
            maxoutsize    = (1,1,1)
        elif len(sys.argv) > 1 and ( "--medium" in sys.argv):
            num_kernels   = [64,64,64]
            kernel_sizes  = [(5, 5), (3, 3), (3,3)]
            maxoutsize    = (1,1,1)
        elif len(sys.argv) > 1 and ( "--large" in sys.argv):
            num_kernels   = [64,64,128]
            kernel_sizes  = [(5, 5), (3, 3), (3,3)]
            maxoutsize    = (2,2,4)

        if '--synapse' in sys.argv:
            classifier = 'synapse'
        elif '--membrane' in sys.argv:
            classifier = 'membrane'
        elif '--synapse_reg' in sys.argv:
            classifier = 'synapse_reg'
        else:
            print 'Error: Invalid Classifier'
            exit()

        if classifier == 'membrane':
            directory_input      = ['data/train-input']
            directory_labels     = ['data/train-labels']
        elif classifier in ['synapse','synapse_reg']:
            directory_input      = ['data/train-input','data/AC3-input']
            directory_labels     = ['data/train-labels','data/AC3-labels']

        if "--pre-process" in sys.argv:
            print "Generating Train/Test Set..."
            read = Read(in_window_shape, out_window_shape, stride, img_size, classifier, n_train_files, n_test_files, samples_per_image, on_ratio, directory_input, directory_labels, membrane_edges)
            read.generate_data()

        return num_kernels, kernel_sizes, maxoutsize, classifier

    def get_params(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params

    def run(self):

        # Hyper-parameters 
        batch_size       = 30
        epochs           = 100
        in_window_shape  = (64,64)
        out_window_shape = (48,48)
        penatly_factor   = 0.,
        maxoutsize       = (1,1,1)
        stride           = 12

        # Image data
        samples_per_image    = 2000
        n_validation_samples = 2000
        on_ratio             = 0.5
        img_size             = (1024,1024)
        n_train_files        = None
        n_test_files         = 5

        
        # Optimizer data
        optimizer                      = 'RMSprop'
        optimizerData                  = {}
        optimizerData['learning_rate'] = 0.001
        optimizerData['rho']           = 0.9
        optimizerData['epsilon']       = 1e-4

        # Classifier: membrane/synapses
        classifier     = 'membrane'
        membrane_edges = 'WideEdges' #GaussianBlur/WideEdges 

        # GLOBAL CONFIG
        theano.config.floatX = 'float32'

        #Random
        rng              = np.random.RandomState(42)
        rngi             = np.random.RandomState(42)
        
        
        ##### PROCESS COMMAND-LINE ARGS #####
        num_kernels, kernel_sizes, maxoutsize,classifier = self.process_cmd_line_args(in_window_shape,out_window_shape,stride,img_size, classifier, n_train_files, n_test_files, samples_per_image, on_ratio, membrane_edges)

        print 'Loading data ...'

        # load in and process data
        preProcess              = PreProcess(n_validation_samples)
        data                    = preProcess.run()
        train_set_x,train_set_y = data[0],data[3]
        valid_set_x,valid_set_y = data[1],data[4]
        test_set_x,test_set_y   = data[2],data[5]

        train_set_x = theano.shared(train_set_x,borrow=True)
        valid_set_x = theano.shared(valid_set_x,borrow=True)
        test_set_x = theano.shared(test_set_x,borrow=True)

        train_set_y = theano.shared(train_set_y,borrow=True)
        valid_set_y = theano.shared(valid_set_y,borrow=True)
        test_set_y = theano.shared(test_set_y,borrow=True)

        print 'Initializing neural network ...'

        # print error if batch size is to large
        if valid_set_y.eval().size<batch_size:
            print 'Error: Batch size is larger than size of validation set.'

        # compute batch sizes for train/test/validation
        n_train_batches  = train_set_x.eval().shape[0]
        n_test_batches   = test_set_x.eval().shape[0]
        n_valid_batches  = valid_set_x.eval().shape[0]

        # adjust batch size
        while n_test_batches % batch_size != 0:
            batch_size += 1 

        print 'Batch size: ',batch_size

        n_train_batches /= batch_size
        n_test_batches  /= batch_size
        n_valid_batches /= batch_size

        # symbolic variables
        x       = T.matrix('x')        # input image data
        y       = T.matrix('y')        # input label data
        
        cov_net = CovNet(rng, batch_size, num_kernels, kernel_sizes, x, y,in_window_shape,out_window_shape,classifier,maxoutsize = maxoutsize, params = self.params)

        # Initialize parameters and functions
        cost        = cov_net.layer4.negative_log_likelihood(y,penatly_factor) # Cost function
        self.params = cov_net.params                                         # List of parameters
        grads       = T.grad(cost, self.params)                                   # Gradient
        index       = T.lscalar()                                            # Index
        
        # Intialize optimizer
        updates = cov_net.init_optimizer(optimizer, cost, self.params, optimizerData)

        # Shuffling of rows for stochastic gradient
        srng = RandomStreams(seed=234)
        perm = theano.shared(np.arange(train_set_x.eval().shape[0]))
        #perm               = theano.shared(np.random.permutation(np.arange(train_set_x.eval().shape[0])))
        rand = theano.shared(np.arange(train_set_x.eval().shape[0]))

        # acc
        acc = cov_net.layer4.errors
        #acc = cov_net.layer4.F1

        #Validation function
        validate_model = theano.function(
                        [index],
                        acc(y),
                        givens = {
                                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )

        #Test function
        test_model = theano.function(
                    [index],
                    acc(y),
                    givens = {
                            x: test_set_x[index * batch_size: (index + 1) * batch_size],
                            y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # Results
        cost_results = []
        val_results  = []
        time_results = []
        
        # Solver
        try:
            print '... Solving'
            start_time = time.time()    
            for epoch in range(epochs):
                t1 = time.time()
                perm              = srng.shuffle_row_elements(perm)
                train_set_x,train_set_y,train_model = f.flip_rotate(train_set_x,train_set_y,in_window_shape,out_window_shape,perm,index,cost,updates,batch_size,x,y,classifier, GPU=False)
                costs             = [train_model(i) for i in xrange(n_train_batches)]
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                t2 = time.time()

                epoch_cost = np.mean(costs)
                epoch_val  = np.mean(validation_losses)
                epoch_time = (t2-t1)/60.

                cost_results.append(epoch_cost)
                val_results.append(epoch_val)
                time_results.append(epoch_time)

                print "Epoch {}    Training Cost: {:.5}   Validation Error: {:.5}    Time (epoch/total): {:.2} mins".format(epoch + 1, epoch_cost, epoch_val, epoch_time)
        except KeyboardInterrupt:
            print 'Exiting solver ...'
            print ''
        
        # End timer
        end_time = time.time()
        end_epochs = epoch+1

        # Timer information
        number_train_samples = train_set_x.eval().shape[0]
        number_test_samples  = test_set_x.eval().shape[0]
        
        predict = theano.function(inputs=[index], 
                                    outputs=cov_net.layer4.prediction(),
                                    givens = {
                                        x: test_set_x[index * batch_size: (index + 1) * batch_size]
                                        }
                                    )
                                    
        # Plot example of output
        if classifier in ['membrane','synapse']:
            output = np.zeros((0,out_window_shape[0]*out_window_shape[1]))
        elif classifier == 'synapse_reg':
            output = np.zeros((0,1))

        start_test_timer = time.time()
        for i in xrange(n_test_batches):
            output = np.vstack((output,predict(i)))
        stop_test_timer = time.time()

        time_per_test_sample = (stop_test_timer-start_test_timer)/float(number_test_samples)
        print "Prediction time per sample:  ", time_per_test_sample
        
        y_pred = test_set_y.eval()
        y      = output
        
        mean_abs_error = np.mean(np.abs(y_pred-y))
        
        print 'Mean Absolute Error (before averaging): ',mean_abs_error

        from sklearn.metrics import f1_score
        print 'F1 score (before averaging): ',f1_score(y.flatten().astype(np.int32),np.round(y_pred).flatten().astype(np.int32))
        
        if classifier in ['membrane', 'synapse']:
            in_shape = (output.shape[0],in_window_shape[0],in_window_shape[1])
            out_shape = (output.shape[0],out_window_shape[0],out_window_shape[1])
        elif classifier == 'synapse_reg':
            in_shape = (output.shape[0],in_window_shape[0],in_window_shape[1])
            out_shape = (output.shape[0],1)

        output = output.reshape(out_shape)

        results    = np.zeros((3, len(cost_results)))
        results[0] = np.array(cost_results)
        results[1] = np.array(val_results)
        results[2] = np.array(time_results)

        table = np.load('data/table.npy')
        output, y = post.post_process(train_set_x.eval(),train_set_y.eval(),output,test_set_y.eval(),table,img_size,in_window_shape,out_window_shape,classifier)

        mean_abs_error = np.mean(np.abs(output-y))
        f1 = f1_score(y.flatten().astype(np.int32),np.round(output).flatten().astype(np.int32))
        print 'Mean Absolute Error (after averaging): ', mean_abs_error
        print 'F1 score (after averaging): ', f1

        if not os.path.exists('results'):
            os.makedirs('results')

        np.save('results/results.npy',results)
        np.save('results/output.npy',output)
        np.save('results/x.npy',test_set_x.eval().reshape(in_shape))
        np.save('results/y.npy',y)

class Engine(object):

    # --------------------------------------------------------------------------
    def __init__(self, total_n_layers = 5):

        # path to store parameters        
        if not os.path.exists('parameters'):
            os.makedirs('parameters')

        path = 'parameters/params.dat'
        
        if "--load-weights_all" in sys.argv: 
            params = self.load_params(path)
            classifier = ConvNetClassifier(params=params)
        elif "--load-weights_1" in sys.argv: 
            load_n_layers = 1
            params = self.load_params(path)
            for n in xrange(load_n_layers,total_n_layers):
                del params["W"+str(n)]
                del params["b"+str(n)]
            classifier = ConvNetClassifier(params=params)
        elif "--load-weights_2" in sys.argv: 
            load_n_layers = 2
            params = self.load_params(path)
            for n in xrange(load_n_layers,total_n_layers):
                del params["W"+str(n)]
                del params["b"+str(n)]
            classifier = ConvNetClassifier(params=params)
        elif "--load-weights_3" in sys.argv: 
            load_n_layers = 3
            params = self.load_params(path)
            for n in xrange(load_n_layers,total_n_layers):
                del params["W"+str(n)]
                del params["b"+str(n)]
            classifier = ConvNetClassifier(params=params)
        elif "--load-weights_4" in sys.argv: 
            load_n_layers = 4
            params = self.load_params(path)
            for n in xrange(load_n_layers,total_n_layers):
                del params["W"+str(n)]
                del params["b"+str(n)]

            classifier = ConvNetClassifier(params=params)
        else:
            classifier = ConvNetClassifier()

        classifier.run()

        # store parameters
        self.save_params(classifier.get_params(), path)

    # --------------------------------------------------------------------------
    def load_params(self, path):
        f = file(path, 'r')
        obj = cPickle.load(f)
        f.close()
        return obj

    # --------------------------------------------------------------------------
    def save_params(self, obj, path):
        f = file(path, 'wb')
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


if __name__ == "__main__":
    engine = Engine()