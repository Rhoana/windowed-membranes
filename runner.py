import sys
import os
import yaml
import time as time
import numpy as np
import theano
import theano.sandbox.cuda
import theano.tensor as T
import util.post_process as post 
import cPickle
from theano.tensor.shared_randomstreams import RandomStreams

from util.build_train_test_set             import BuildTrainTestSet 
from util.runner_functions                 import RunnerFunctions
from pre_process.pre_process               import Read 
from edge_prediction_conv.edge_cov_net     import CovNet 
from edge_prediction_conv.helper_functions import Functions as f 

class ConvNetClassifier(RunnerFunctions):
    
    def __init__(self,params = {}):
        self.params = params

        if not os.path.exists("parameters"):
            os.makedirs("parameters")

    def run(self):

        if self.pre_process:
            self.generate_train_test_set(config_file)
            if self.pre_process_only:
                sys.exit(0)

        if self.load_n_layers != -1:
            self.load_layers(self.load_n_layers)

         #Random
        rng              = np.random.RandomState(42)
        rngi             = np.random.RandomState(42)
        
        print 'Loading data ...'

        # load in and process data
        preProcess              = BuildTrainTestSet(self.n_validation_samples,self.pre_processed_folder)
        data                    = preProcess.run(self.classifier,config_file)
        
        if self.predict_only == False:
            train_set_x,train_set_y = data[0],data[3]
            n_train_batches  = train_set_x.get_value(borrow=True).shape[0]

        valid_set_x,valid_set_y = data[1],data[4]
        n_valid_batches  = valid_set_x.get_value(borrow=True).shape[0]
        test_set_x,test_set_y   = data[2],data[5]
        n_test_batches   = test_set_x.get_value(borrow=True).shape[0]

        print 'Initializing neural network ...'

        # print error if batch size is to large
        if valid_set_y.eval().size<self.batch_size:
            print 'Error: Batch size is larger than size of validation set.'

        
        # adjust batch size
        while n_test_batches % self.batch_size != 0:
            self.batch_size += 1 

        print 'Batch size: ',self.batch_size
        n_train_batches /= self.batch_size
        n_test_batches  /= self.batch_size
        n_valid_batches /= self.batch_size

        # symbolic variables
        x       = T.matrix('x')        # input image data
        y       = T.matrix('y')        # input label data
        
        cov_net = CovNet(rng, self.batch_size, self.layers_3D, self.num_kernels, self.kernel_sizes, x, y,self.in_window_shape,self.out_window_shape,self.classifier,maxoutsize = self.maxoutsize, params = self.params, dropout = self.dropout)

        cov_net_test = cov_net.TestVersion(rng, self.batch_size, self.layers_3D, self.num_kernels, self.kernel_sizes, x, y,self.in_window_shape,self.out_window_shape,self.classifier,maxoutsize = self.maxoutsize, params = self.params, network = cov_net, dropout = [0.,0.,0.,0.])

        # Initialize parameters and functions
        cost        = cov_net.layer4.negative_log_likelihood(y,self.penalty_factor) # Cost function
        self.params = cov_net.params                                         # List of parameters
        grads       = T.grad(cost, self.params)                                   # Gradient
        index       = T.lscalar()                                            # Index
        
        # Intialize optimizer
        updates = cov_net.init_optimizer(self.optimizer, cost, self.params, self.optimizerData)

        # Shuffling of rows for stochastic gradient
        srng = RandomStreams(seed=234)
        perm = theano.shared(np.arange(train_set_x.eval().shape[0]))
        rand = theano.shared(np.arange(train_set_x.eval().shape[0]))

        # acc
        errors = cov_net_test.layer4_test.errors
        
        # Train model   
        if self.predict_only == False:
            train_model = theano.function(                                          
                        [index],                                                    
                            cost,                                                       
                            updates = updates,                                          
                            givens  = {                                                 
                                        x: train_set_x[perm[index * self.batch_size: (index + 1) * self.batch_size]], 
                                        y: train_set_y[perm[index * self.batch_size: (index + 1) * self.batch_size]]
                }                                                                   
            )

            #Validation function
            validate_model = theano.function(
                            [index],
                            errors(y),
                            givens = {
                                    x: valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                                    y: valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
                    }
                )

        #Test function
        test_model = theano.function(
                    [index],
                    errors(y),
                    givens = {
                            x: test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                            y: test_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        if self.predict_only == False:

            # Results
            cost_results = []
            val_results  = []
            time_results = []
            
            # Solver
            try:
                print '... Solving'
                start_time = time.time()    
                for epoch in range(self.epochs):
                    t1 = time.time()
                    perm              = srng.shuffle_row_elements(perm)
                    train_set_x,train_set_y = f.flip_rotate(train_set_x,train_set_y,self.in_window_shape,self.out_window_shape,perm,index,cost,updates,self.batch_size,x,y,self.classifier,self.layers_3D)
                    costs             = [train_model(i) for i in xrange(n_train_batches)]
                    validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                    t2 = time.time()

                    epoch_cost = np.mean(costs)
                    epoch_val  = np.mean(validation_losses)
                    epoch_time = (t2-t1)/60.

                    cost_results.append(epoch_cost)
                    val_results.append(epoch_val)
                    time_results.append(epoch_time)

                    # store parameters
                    self.save_params(self.get_params(), self.path)

                    print "Epoch {}    Training Cost: {:.5}   Validation Error: {:.5}    Time (epoch/total): {:.2} mins".format(epoch + 1, epoch_cost, epoch_val, epoch_time)
            except KeyboardInterrupt:
                print 'Exiting solver ...'
                print ''
            
            # End timer
            end_time = time.time()
            end_epochs = epoch+1

        # Timer information
        number_train_samples = train_set_x.get_value(borrow=True).shape[0]
        number_test_pixels  = test_set_y.get_value(borrow=True).shape[0]*test_set_y.get_value(borrow=True).shape[1]
        
        predict = theano.function(inputs=[index], 
                                    outputs=cov_net.layer4.prediction(),
                                    givens = {
                                        x: test_set_x[index * self.batch_size: (index + 1) * self.batch_size]
                                        }
                                    )
                                    
        # Plot example of output
        if self.classifier in ['membrane','synapse']:
            output = np.zeros((0,self.out_window_shape[0]*self.out_window_shape[1]))
        elif self.classifier == 'synapse_reg':
            output = np.zeros((0,1))

        start_test_timer = time.time()
        for i in xrange(n_test_batches):
            output = np.vstack((output,predict(i)))
        stop_test_timer = time.time()

        pixels_per_second = number_test_pixels/(stop_test_timer-start_test_timer)
        print "Prediction: pixels per second:  ", pixels_per_second
        
        mean_abs_error = np.mean(np.abs(output-test_set_y.get_value(borrow=True)))
        
        print 'Mean Absolute Error (before averaging): ',mean_abs_error

        if self.classifier in ['membrane', 'synapse']:
            out_shape = (output.shape[0],self.out_window_shape[0],self.out_window_shape[1])
        elif self.classifier == 'synapse_reg':
            out_shape = (output.shape[0],1)
        output = output.reshape(out_shape)

        results    = np.zeros((3, len(cost_results)))
        results[0] = np.array(cost_results)
        results[1] = np.array(val_results)
        results[2] = np.array(time_results)

        table = np.load(self.pre_processed_folder + 'table.npy')
        output, y = post.post_process(train_set_x.get_value(borrow=True),train_set_y.get_value(borrow=True),output,test_set_y.get_value(borrow=True),table,self.img_size,self.in_window_shape,self.out_window_shape,self.classifier)

        mean_abs_error = np.mean(np.abs(output-y))
        print 'Mean Absolute Error (after averaging): ', mean_abs_error
        
        np.save(self.results_folder + 'results.npy', results)
        np.save(self.results_folder + 'output.npy', output)
        np.save(self.results_folder + 'y.npy', y)

        latest_run = open('latest_run.txt', 'w')
        latest_run.write(self.ID_folder + "\n")
        latest_run.close()




if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "default.yaml"
    conv_net_classifier = ConvNetClassifier()
    conv_net_classifier.init(config_file)
    conv_net_classifier.run()
