import os
import numpy as np
import yaml
import cPickle
import theano
import time
from scipy import signal
import mahotas as mh

try:
    import partition_comparison
    evaluate_VI = True
except:
    print "Unable to load package partition_comparison (needed for Variation of Information)"
    evaluate_VI = False

from pre_process.pre_process               import Read 

class RunnerFunctions(object):
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

    # --------------------------------------------------------------------------
    def load_layers(self, load_n_layers):
        if self.load_n_layers != -1:
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
        else:
            pass

    # --------------------------------------------------------------------------
    def generate_train_test_set(self, config_file):
        if self.pre_process == True:
            print "Generating Train/Test Set..."
            read = Read(self.in_window_shape, 
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
                    self.adaptive_histogram_equalization,
                    self.pre_processed_folder,
                    self.predict_only,
                    self.predict_train_set,
                    self.images_from_numpy)

            read.generate_data(config_file)
        else:
            pass

    # --------------------------------------------------------------------------
    def get_params(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params

    # --------------------------------------------------------------------------
    def get_config(self, custom_config_file):
        global_config = open("config/global.yaml")
        global_data_map = yaml.safe_load(global_config)
        custom_config = open("config/" + custom_config_file)
        custom_data_map = yaml.safe_load(custom_config)
        return global_data_map, custom_data_map

    # --------------------------------------------------------------------------
    def get_locals(self, global_data_map, custom_data_map):
        for key in ["hyper-parameters", "image-data", "optimizer-data", "classifier", "weights", "main","evaluation"]:
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
        locals().update(global_data_map["network"][custom_data_map["network"]["size"]])
        self.size = custom_data_map["network"]["size"]
        # set training data location
        locals().update(global_data_map["network"]["training-data"][custom_data_map["classifier"]["classifier"]])
        # load weights
        locals().update(custom_data_map["weights"])
        self.path = global_data_map["weights"]["weights_path"]
        
        for key, value in locals().iteritems():
            if key not in ["global_data_map", "custom_data_map", "data_map", "self"]:
                setattr(self, key, value)
        return True

    # --------------------------------------------------------------------------
    def init(self,config_file):
        global_data_map, custom_data_map = self.get_config(config_file)
        self.get_locals(global_data_map, custom_data_map)
        self.define_folders()

        #QUICK-FIX
        self.num_kernels  = (self.num_kernels[0],self.num_kernels[1],self.num_kernels[2])
        self.kernel_sizes = ((self.kernel_sizes[0][0],self.kernel_sizes[0][1]),(self.kernel_sizes[1][0],self.kernel_sizes[1][1]),(self.kernel_sizes[2][0],self.kernel_sizes[2][1]))
        self.maxoutsize = (self.maxoutsize[0],self.maxoutsize[1],self.maxoutsize[2])

    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    def init_predict(self,data_set,net,batch_size,x,index):
        predict = theano.function(inputs=[index], 
                                outputs=net.layer4.prediction(),
                                givens = {
                                    x: data_set[index * self.batch_size: (index + 1) * self.batch_size]
                                    }
                                )
        return predict

    # --------------------------------------------------------------------------
    def predict_set(self,predict,n_batches,number_pixels = None):
                                
        output = np.zeros((0,self.pred_window_size[1]**2))

        start_test_timer = time.time()
        for i in xrange(n_batches):
            output = np.vstack((output,predict(i)))
        stop_test_timer = time.time()

        if number_pixels != None:
            pixels_per_second = number_pixels/(stop_test_timer-start_test_timer)
            print "Prediction, pixels per second:  ", pixels_per_second

        return output

    # --------------------------------------------------------------------------
    def evaluate(self,pred,ground_truth,eval_window_size = None):
        eval_window_size = self.eval_window_size

        ground_truth = ground_truth[:pred.shape[0]]
        eval_window = np.ones((eval_window_size,eval_window_size))

        # Reshape
        if ground_truth.ndim == 2:
            ground_truth = ground_truth.reshape(ground_truth.shape[0],int(np.sqrt(ground_truth.shape[1])),int(np.sqrt(ground_truth.shape[1])))
            pred = pred.reshape(ground_truth.shape)

        # Calculate pixel-wise error
        pixel_error = np.mean(np.abs(pred-ground_truth))

        # Calculate window-wise error
        windowed_error = 0
        for n in xrange(pred.shape[0]):
            pred_conv         = signal.convolve2d(pred[n],eval_window,mode="valid")/float(eval_window_size**2)
            ground_truth_conv = signal.convolve2d(ground_truth[n],eval_window,mode="valid")/float(eval_window_size**2)
            pred_conv          = pred_conv[::eval_window_size,::eval_window_size]
            ground_truth_conv  = ground_truth_conv[::eval_window_size,::eval_window_size]

            windowed_error += np.mean(np.abs(pred_conv-ground_truth_conv))

        windowed_error = windowed_error/float(pred.shape[0])

        return pixel_error,windowed_error

    # --------------------------------------------------------------------------
    def evaluate_F1_watershed(self,pred):
        
        V1_metric = None 
        F1_pixel  = None
        F1_window = None
        
        if self.classifier == "membrane" and evaluate_VI == True:
            def disk(radius):
                y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
                return x**2 + y**2 <= radius**2

    
            def watershed(probs, radius):
                sel = disk(radius)
                minima = mh.regmin(probs, Bc=sel)
                markers, nr_markers = mh.label(minima)
                return mh.cwatershed(probs, markers)

            def VI(putative, gold):
                mask = gold != 0
                VI_res = partition_comparison.variation_of_information(gold[mask], putative[mask].astype(gold.dtype))
                return VI_res
            
            # Load in test-addresses
            f = file(self.pre_processed_folder + "test_adress.dat", 'rb')
            test_adress = cPickle.load(f)
            f.close()
        
            gap = (self.in_window_shape[0]-self.pred_window_size[1])/2
            VI_metric = 0
            n = 0
            for adress in test_adress:
                ground_truth = mh.imread(adress)[gap:-gap,gap:-gap]
            
                radii = np.linspace(7, 65, 10) 
                ground_truth = ground_truth.astype(np.uint16)
                VIs = np.array([VI(watershed(pred[n], radius), ground_truth) for radius in radii])
                VI_metric += np.min(VIs)
            
                n += 1 
            
            VI_metric /= n 
        
        elif self.classifier == "synapse":
            pass
        
        
        return VI_metric, F1_pixel, F1_window
        

    # --------------------------------------------------------------------------
    def write_last_run(self,ID_folder):
        latest_run = open('latest_run.txt', 'w')
        latest_run.write(ID_folder + "\n")
        latest_run.close()

    # --------------------------------------------------------------------------
    def write_results(self, 
            error_pixel_before, 
            error_window_before, 
            error_pixel_after, 
            error_window_after,
            VI,
            F1_pixel,
            F1_window):

        results_file = open(self.results_folder + "results.txt", "w")

        results_file.write("Pixel-error before averaging: " + str(error_pixel_before) + "\n")
        results_file.write("Window-error before averaging: " + str(error_window_before) + "\n")
        results_file.write("Pixel-error after averaging: " + str(error_pixel_after) + "\n")
        results_file.write("Window-error after averaging: " + str(error_window_after) + "\n")
        
        if VI != None:
            results_file.write("Variation of information: " + str(VI) + "\n")
        if F1_pixel != None:
            results_file.write("F1-score, pixel: "  + str(F1_pixel) + "\n")
        if F1_window != None:
            results_file.write("F1-score, window: " + str(f1_window) + "\n")

        results_file.close()

    # --------------------------------------------------------------------------
    def write_parameters(self,epochs,n_train_samples):
        parameter_file = open(self.results_folder + "parameters.txt", "w")

        parameter_file.write("Classifier: " + str(self.classifier) + "\n")
        parameter_file.write("Network size: " + str(self.size) + "\n")
        parameter_file.write("Number of epochs: " + str(epochs) + "\n")
        parameter_file.write("Prediction window size " + str(','.join(str(v) for v in self.pred_window_size) + "\n"))
        parameter_file.write("Stride length: " + str(self.stride) + "\n")
        parameter_file.write("Penalty factor: " + str(self.penalty_factor) + "\n")
        parameter_file.write("Layers_3D: " + str(self.layers_3D) + "\n")
        parameter_file.write("Number of training samples: " + str(n_train_samples) + "\n")
        parameter_file.write("Samples per training image: " + str(self.samples_per_image) + "\n")
        parameter_file.write("Number of test files: " + str(self.n_test_files) + "\n")

        parameter_file.close()







