import sys
import os
import yaml
import time as time
import numpy as np
import theano
import theano.sandbox.cuda
import theano.tensor as T
import cPickle
from theano.tensor.shared_randomstreams import RandomStreams

import post_process.post_process as post 
from util.build_train_test_set             import BuildTrainTestSet 
from util.runner_functions                 import RunnerFunctions
from pre_process.pre_process               import Read 
from models.conv_net     import ConvNet 
from util.helper_functions import Functions as f 

class ConvNetClassifier(RunnerFunctions):
    
    def __init__(self,params = {}):
        self.params = params

        if not os.path.exists("parameters"):
            os.makedirs("parameters")

    def run(self):

        # Generate training set and test set
        self.generate_train_test_set(config_file)

        # Load weight layers
        self.load_layers(self.load_n_layers)

        # Initialize random stream
        rng              = np.random.RandomState(42)
        
        print 'Loading data ...'

        # load in and pre-process data
        preProcess              = BuildTrainTestSet(self.n_validation_samples,self.pre_processed_folder)
        data                    = preProcess.build_train_val_set()
        data2                   = preProcess.build_test_set()
        
        if self.predict_only == False:
            train_set_x,train_set_y = data[0],data[2]
            n_train_batches         = train_set_x.get_value(borrow=True).shape[0]

        valid_set_x,valid_set_y = data[1],data[3]
        n_valid_batches         = valid_set_x.get_value(borrow=True).shape[0]
        test_set_x,test_set_y   = data2[0],data2[1]
        n_test_batches          = test_set_x.get_value(borrow=True).shape[0]

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
        
        # Initialize networks
        conv_net = ConvNet(rng, 
                self.batch_size, 
                self.layers_3D, 
                self.num_kernels, 
                self.kernel_sizes, 
                x, 
                y,
                self.in_window_shape,
                self.out_window_shape,
                self.classifier,
                maxoutsize = self.maxoutsize, 
                params = self.params, 
                dropout = self.dropout)

        conv_net_test = conv_net.TestVersion(rng, 
                self.batch_size, 
                self.layers_3D, 
                self.num_kernels, 
                self.kernel_sizes, 
                x, 
                y,
                self.in_window_shape,
                self.out_window_shape,
                self.classifier,
                maxoutsize = self.maxoutsize, 
                params = self.params, 
                network = conv_net, 
                dropout = [0.,0.,0.,0.5])

        # Initialize parameters and functions
        cost        = conv_net.layer4.negative_log_likelihood(y,self.penalty_factor) # Cost function
        self.params = conv_net.params                                                # List of parameters
        grads       = T.grad(cost, self.params)                                     # Gradient
        index       = T.lscalar()                                                   # Index
        
        # Intialize optimizer
        updates = conv_net.init_optimizer(self.optimizer, cost, self.params, self.optimizerData)
        srng = RandomStreams(seed=234)
        perm = theano.shared(np.arange(train_set_x.eval().shape[0]))

        # Train functions
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


            # Initialize result arrays
            cost_results = []
            val_results  = []
            time_results = []

            predict_val = self.init_predict(valid_set_x,conv_net_test,self.batch_size,x,index)

            # Solver
            try:
                print '... Solving'
                start_time = time.time()    
                for epoch in range(self.epochs):
                    t1 = time.time()
                    perm              = srng.shuffle_row_elements(perm)
                    train_set_x,train_set_y = f.flip_rotate(train_set_x,
                            train_set_y,
                            self.in_window_shape,
                            self.out_window_shape,
                            perm,
                            index,
                            cost,
                            updates,
                            self.batch_size,
                            x,
                            y,
                            self.classifier,
                            self.layers_3D)

                    costs             = [train_model(i) for i in xrange(n_train_batches)]
                    epoch_cost = np.mean(costs)
                    output_val = self.predict_set(predict_val,n_valid_batches)
                    epoch_val  = self.evaluate(output_val,valid_set_y.get_value(borrow=True))
                    
                    t2 = time.time()
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

        #results    = np.zeros((3, len(cost_results)))
        #results[0] = np.array(cost_results)
        #results[1] = np.array(val_results)
        #results[2] = np.array(time_results)
        #np.save(self.results_folder + 'results.npy', results)
        
        # Timer information
        number_test_pixels  = test_set_y.get_value(borrow=True).shape[0]*test_set_y.get_value(borrow=True).shape[1]
        
        # Predict test set
        predict_test = self.init_predict(test_set_x,conv_net_test,self.batch_size,x,index)
        output       = self.predict_set(predict_test,n_test_batches,number_pixels=number_test_pixels)

        # Evaluate error
        error = self.evaluate(output,test_set_y.get_value(borrow=True))
        print 'Error after averaging: ', error

        # Post-process
        table = np.load(self.pre_processed_folder + 'table.npy')
        output = output.reshape((output.shape[0],self.out_window_shape[0],self.out_window_shape[1]))
        output, y = post.post_process(train_set_x.get_value(borrow=True),
                train_set_y.get_value(borrow=True),
                output,
                test_set_y.get_value(borrow=True),
                table,
                self.img_size,
                self.in_window_shape,
                self.out_window_shape,
                self.classifier)

        # Evalute error after post-processing 
        error = self.evaluate(output,y)
        print 'Error after averaging: ', error
        
        # Save and write
        np.save(self.results_folder + 'output.npy', output)
        np.save(self.results_folder + 'y.npy', y)
        self.write_last_run(self.ID_folder)


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "default.yaml"
    conv_net_classifier = ConvNetClassifier()
    conv_net_classifier.init(config_file)
    conv_net_classifier.run()
