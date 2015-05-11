import sys
import yaml
import time as time
import os
import numpy as np
import theano
import theano.sandbox.cuda
import theano.tensor as T
import util.post_process as post 
import cPickle
from theano.tensor.shared_randomstreams import RandomStreams

from util.build_train_test_set          import BuildTrainTestSet 
from pre_process.pre_process            import Read 
from edge_prediction_conv.edge_cov_net import CovNet 
from edge_prediction_conv.helper_functions import Functions as f 

class ConvNetClassifier(object):
    
    def __init__(self,params = {}):
        self.params = params

        if not os.path.exists("parameters"):
            os.makedirs("parameters")


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

    def load_layers(self, load_n_layers):
        total_n_layers = 5
        if os.path.isfile(self.path) == True:
            params = self.load_params(self.path)
            self.params = params
            for n in xrange(load_n_layers,total_n_layers):
                del self.params["W"+str(n)]
                del self.params["b"+str(n)]
        else:
            self.params = None
            print 'Warning: Unable to load weights'
        return True

    def generate_train_test_set(self, config_file):
        print "Generating Train/Test Set..."
        read = Read(self.in_window_shape, self.out_window_shape, self.stride, self.img_size, self.classifier, self.n_train_files, self.n_test_files, self.samples_per_image, self.on_ratio, self.directory_input, self.directory_labels, self.membrane_edges,self.layers_3D, self.adaptive_histogram_equalization,self.pre_processed_folder,self.predict_only,self.predict_train_set,self.images_from_numpy)
        read.generate_data(config_file)
        return True

    def get_params(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params

    def get_config(self, custom_config_file):
        global_config = open("config/global.yaml")
        global_data_map = yaml.safe_load(global_config)
        custom_config = open("config/" + custom_config_file)
        custom_data_map = yaml.safe_load(custom_config)
        return global_data_map, custom_data_map

    def get_locals(self, global_data_map, custom_data_map):
        for key in ["hyper-parameters", "image-data", "optimizer-data", "classifier", "weights_path", "pre-process", "main"]:
            locals().update(global_data_map[key])
            if key in custom_data_map:
                locals().update(custom_data_map[key])

        for data_map in [global_data_map, custom_data_map]:
            if 'theano' in data_map:
                for key, value in data_map['theano']['config'].iteritems():
                    if key == "device":
                        theano.sandbox.cuda.use(value)
                    else:
                        setattr(theano.config, key, value)

        # set convolution size
        locals().update(global_data_map["convolution"][custom_data_map["convolution"]["size"]])
        # set training data location
        locals().update(global_data_map["convolution"]["training-data"][custom_data_map["classifier"]["classifier"]])
        # load_n_layers  
        locals().update(custom_data_map["load-weights"])
        # load weights_path
        locals().update(global_data_map["weights_path"])
        self.path = global_data_map["weights_path"]["weights_path"]
        
        for key, value in locals().iteritems():
            if key not in ["global_data_map", "custom_data_map", "data_map", "self"]:
                setattr(self, key, value)
        return True

    def init(self,config_file):
        global_data_map, custom_data_map = self.get_config(config_file)
        self.get_locals(global_data_map, custom_data_map)
        self.define_folders()

        #QUICK-FIX
        self.num_kernels  = (self.num_kernels[0],self.num_kernels[1],self.num_kernels[2])
        self.kernel_sizes = ((self.kernel_sizes[0][0],self.kernel_sizes[0][1]),(self.kernel_sizes[1][0],self.kernel_sizes[1][1]),(self.kernel_sizes[2][0],self.kernel_sizes[2][1]))
        self.maxoutsize = (self.maxoutsize[0],self.maxoutsize[1],self.maxoutsize[2])

    def define_folders(self):
        self.ID_folder = "run_data/" + self.ID_folder

        if self.pre_process == True: 
            folder_exists = os.path.isdir(self.ID_folder)
            original_ID_folder = self.ID_folder
            n = 0
            while folder_exists:
                self.ID_folder = original_ID_folder + "_" + str(n)
                folder_exists = os.path.isdir(self.ID_folder)
                n +=1

            if n > 0:
                print "Warning: Changed folder ID"

            self.pre_processed_folder = self.ID_folder + "/pre_processed/"
            self.results_folder = self.ID_folder + "/results/"

            if os.path.isdir(self.pre_processed_folder) != True:
                os.makedirs(self.pre_processed_folder)
            if os.path.isdir(self.results_folder) != True:
                os.makedirs(self.results_folder)

        else:
            try:
                self.pre_processed_folder = self.ID_folder + "/pre_processed/"
                self.results_folder = self.ID_folder + "/results/"
            except:
                print "Error: Unable to find pre-processed data"
                exit()

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
        
        cov_net = CovNet(rng, self.batch_size, self.layers_3D, self.num_kernels, self.kernel_sizes, x, y,self.in_window_shape,self.out_window_shape,self.classifier,maxoutsize = self.maxoutsize, params = self.params)

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
        #perm               = theano.shared(np.random.permutation(np.arange(train_set_x.eval().shape[0])))
        rand = theano.shared(np.arange(train_set_x.eval().shape[0]))

        # acc
        acc = cov_net.layer4.errors
        #acc = cov_net.layer4.F1
        
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
                            acc(y),
                            givens = {
                                    x: valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                                    y: valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
                    }
                )

        #Test function
        test_model = theano.function(
                    [index],
                    acc(y),
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
