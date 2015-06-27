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

from util.runner_functions                 import RunnerFunctions
from util.worker                           import Worker
from models.conv_net                       import ConvNet
from models.fully_con                      import FullyCon,FullyConCompressed
from util.optimizer                        import Optimizer
import util.helper_functions as f 

from layers.in_layer import InLayer

rng              = np.random.RandomState(42)

class ConvNetClassifier(RunnerFunctions):
    def __init__(self,params = {}):
        self.params = params

        if not os.path.exists("parameters"):
            os.makedirs("parameters")

    def run(self):
        
        worker = Worker(self.in_window_shape, 
                self.out_window_shape, 
                self.pred_window_size,
                self.stride, 
                self.img_size, 
                self.classifier, 
                self.n_train_files, 
                self.n_test_files, 
                self.samples_per_image, 
                self.on_ratio, 
                self.directory_input, 
                self.directory_labels, 
                self.membrane_edges,
                self.layers_3D, 
                self.pre_processed_folder,
                self.batch_size,
                self.num_kernels,
                self.kernel_sizes,
                self.maxoutsize,
                self.params,
                self.eval_window_size,
                config_file,
                self.n_validation_samples)

        if self.pre_process == True:
            print "Generating Train/Test Set..."

            worker.generate_train_data()
        
        data,n_train_samples = worker.get_train_data()

        # Load weight layers
        self.load_layers(self.load_n_layers)
        
        print 'Loading data ...'
        
        if self.predict_only == False:
            train_set_x,train_set_y = data[0],data[2]
            n_train_batches         = train_set_x.get_value(borrow=True).shape[0]

        valid_set_x,valid_set_y = data[1],data[3]
        n_valid_batches         = valid_set_x.get_value(borrow=True).shape[0]

        print 'Initializing neural network ...'

        # print error if batch size is to large
        if valid_set_y.eval().size<self.batch_size:
            print 'Error: Batch size is larger than size of validation set.'
        
        print 'Batch size: ',self.batch_size

        n_train_batches /= self.batch_size
        n_valid_batches /= self.batch_size

        # symbolic variables
        x       = T.matrix('x')        # input image data
        y       = T.matrix('y')        # input label data
        
        # Initialize networks
        if self.net == 'ConvNet':
            model = ConvNet(rng, 
                    self.batch_size, 
                    self.layers_3D, 
                    self.num_kernels, 
                    self.kernel_sizes, 
                    x, 
                    y,
                    self.in_window_shape,
                    self.out_window_shape,
                    self.pred_window_size,
                    self.classifier,
                    maxoutsize = self.maxoutsize, 
                    params = self.params, 
                    dropout = self.dropout)

            model_val = model.TestVersion(rng, 
                    self.batch_size, 
                    self.layers_3D, 
                    self.num_kernels, 
                    self.kernel_sizes, 
                    x, 
                    y,
                    self.in_window_shape,
                    self.out_window_shape,
                    self.pred_window_size,
                    self.classifier,
                    maxoutsize = self.maxoutsize, 
                    params = self.params, 
                    network = model, 
                    dropout = [0.,0.,0.,0.0])
                
        elif self.net == "FullyCon":
            model = FullyCon(rng, 
                    self.batch_size, 
                    self.layers_3D, 
                    x, 
                    y,
                    self.in_window_shape,
                    self.out_window_shape,
                    self.pred_window_size,
                    self.classifier,
                    params = self.params)

            model_val= model.TestVersion(rng, 
                    self.batch_size, 
                    self.layers_3D, 
                    x, 
                    y,
                    self.in_window_shape,
                    self.out_window_shape,
                    self.pred_window_size,
                    self.classifier,
                    params = self.params, 
                    network = model)
        elif self.net == "FullyConCompressed":
            model = FullyConCompressed(rng, 
                    self.batch_size, 
                    self.layers_3D, 
                    x, 
                    y,
                    self.in_window_shape,
                    self.out_window_shape,
                    self.pred_window_size,
                    self.classifier,
                    params = self.params)

            model_val= model.TestVersion(rng, 
                    self.batch_size, 
                    self.layers_3D, 
                    x, 
                    y,
                    self.in_window_shape,
                    self.out_window_shape,
                    self.pred_window_size,
                    self.classifier,
                    params = self.params, 
                    network = model)
        else:
            raise RuntimeError('Unable to load network: ' + str(self.net))

        # Initialize parameters and functions
        cost        = model.layer4.negative_log_likelihood(self.penalty_factor)  # Cost function
        self.params = model.params                                               # List of parameters
        grads       = T.grad(cost, self.params)                                     # Gradient
        index       = T.lscalar()                                                   # Index
        
        # Intialize optimizera
        optimizer = Optimizer()
        updates = optimizer.init_optimizer(self.optimizer, cost, self.params, self.optimizerData)
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
            cost_results        = []
            val_results_pixel   = []
            time_results        = []

            predict_val = f.init_predict(valid_set_x, model_val,self.batch_size,x,index)

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
                    output_val = f.predict_set(predict_val,n_valid_batches,self.classifier, self.pred_window_size)
                    error_pixel,error_window = f.evaluate(output_val,valid_set_y.get_value(borrow=True),self.eval_window_size,self.classifier)
                    #error_pixel = 0.
                    #error_window = 0.

                    t2 = time.time()
                    epoch_time = (t2-t1)/60.

                    cost_results.append(epoch_cost)
                    val_results_pixel.append(error_pixel)
                    time_results.append(epoch_time)

                    # store parameters
                    self.save_params(self.get_params(), self.path)

                    if self.classifier in ["membrane","synapse"]:
                        print "Epoch {}    Training Cost: {:.5}   Validation Error (pixel/window): {:.5}/{:.5}    Time (epoch/total): {:.2} mins".format(epoch + 1, epoch_cost, error_pixel,error_window, epoch_time)
                    else:
                        print "Epoch {}    Training Cost: {:.5}   Validation Error, Membrane (pixel/window): {:.5}/{:.5}    Validation Error, Synapse (pixel/window): {:.5}/{:.5}   Time (epoch/total): {:.2} mins".format(epoch + 1, epoch_cost, error_pixel[0],error_window[0],error_pixel[1],error_window[1], epoch_time)
            except KeyboardInterrupt:
                print 'Exiting solver ...'
                print ''
            
            # End timer
            end_time = time.time()
            end_epochs = epoch+1

        try:
            results    = np.zeros((4, len(cost_results)))
            results[0] = np.array(cost_results)
            results[1] = np.array(val_results_pixel)
            results[2] = np.array(time_results)
            np.save(self.results_folder + 'results.npy', results)
        except:
            pass
            
        (output, 
         y, 
         error_pixel_before, 
         error_window_before, 
         error_pixel_after, 
         error_window_after) = worker.generate_test_data(model,x,y,index,self.net)

        print 'Error before averaging (pixel/window): ' + str(error_pixel_before) + "/" + str(error_window_before)
        print 'Error after averaging (pixel/window): ' + str(error_pixel_after) + "/" + str(error_window_after)

        # Save and write
        self.write_results(error_pixel_before,error_window_before,error_pixel_after,error_window_after)
        self.write_parameters(end_epochs,n_train_samples)
        
        np.save(self.results_folder + 'output.npy', output)
        np.save(self.results_folder + 'y.npy', y)
        self.write_last_run(self.ID_folder)
        
        make_set = True
        if make_set == True:
            from PIL import Image
            set_folder = self.results_folder+"/set/"
            if os.path.isdir(set_folder) != True:
                os.makedirs(set_folder)
     
            for n in xrange(output.shape[0]):
                im = Image.fromarray(np.uint8(output[n]*255))
                im.save(set_folder + "set_" + str(n) + ".tif")
            
            

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "default.yaml"
    conv_net_classifier = ConvNetClassifier()
    conv_net_classifier.init(config_file)
    conv_net_classifier.run()




