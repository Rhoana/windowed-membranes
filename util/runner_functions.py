import os
import numpy as np
import yaml
import cPickle
import theano

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
        for key in ["hyper-parameters", "image-data", "optimizer-data", "classifier", "weights", "main"]:
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
        # set training data location
        locals().update(global_data_map["network"]["training-data"][custom_data_map["classifier"]["classifier"]])
        # load weights
        locals().update(custom_data_map["weights"])
        self.path = global_data_map["weights"]["weights_path"]
        
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
