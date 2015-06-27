import theano
import theano.tensor as T
import numpy as np
import time
from scipy import signal

from util.utils import *
from collections import OrderedDict

rng = np.random.RandomState(42) 
srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))  

def flip_rotate(train_set_x,train_set_y,in_window_shape,out_window_shape,perm,index,cost,updates,batch_size,x,y,classifier,layers_3D):
    temp_x     = train_set_x.get_value(borrow = True)
    temp_y     = train_set_y.get_value(borrow = True)

    if classifier in ['membrane','synapse']:
        temp_x     = temp_x.reshape(temp_x.shape[0],layers_3D,in_window_shape[0],in_window_shape[1])
        temp_y     = temp_y.reshape(temp_y.shape[0],out_window_shape[0],out_window_shape[1])

        n_temp_x   = temp_x.shape[0]
        flip1_prob = 0.5
        flip1_n    = int(np.floor(flip1_prob*n_temp_x))
        flip2_prob = 0.5
        flip2_n    = int(np.floor(flip2_prob*n_temp_x))
        rot_prob   = 0.5
        rot_n      = int(np.floor(rot_prob*n_temp_x))
        
        perm1 = np.random.permutation(range(n_temp_x))[:flip1_n]
        perm2 = np.random.permutation(range(n_temp_x))[:flip2_n]
        perm3 = np.random.permutation(range(n_temp_x))[:rot_n]

        for n in xrange(flip1_n):
            temp_x[perm1[n]] = temp_x[perm1[n],:,::-1,:]
            temp_y[perm1[n]] = temp_y[perm1[n],::-1,:]

        for n in xrange(flip2_n):
            temp_x[perm2[n]] = temp_x[perm2[n],:,:,::-1]
            temp_y[perm2[n]] = temp_y[perm2[n],:,::-1]

        ######### NEED TO FIX ROTATIONS!!!!!!!!!!!!!!!!
        for n in xrange(flip2_n):
            rand = np.random.randint(1,4)

            for m in xrange(temp_x.shape[1]):
                temp_x[perm2[n],m] = np.rot90(temp_x[perm2[n],m],rand)
            temp_y[perm2[n]] = np.rot90(temp_y[perm2[n]],rand)

        temp_x = temp_x.reshape(temp_x.shape[0],layers_3D*temp_x.shape[2]**2)
        temp_y = temp_y.reshape(temp_y.shape[0],temp_y.shape[1]**2)


    elif classifier == 'synapse_reg':
        temp_x     = temp_x.reshape(temp_x.shape[0],layers_3D,in_window_shape[0],in_window_shape[1])
        temp_y     = temp_y.reshape(temp_y.shape[0],1)
        
        n_temp_x   = temp_x.shape[0]
        flip1_prob = 0.5
        flip1_n    = int(np.floor(flip1_prob*n_temp_x))
        flip2_prob = 0.5
        flip2_n    = int(np.floor(flip2_prob*n_temp_x))
        rot_prob   = 0.5
        rot_n      = int(np.floor(rot_prob*n_temp_x))
        
        perm1 = np.random.permutation(range(n_temp_x))[:flip1_n]
        perm2 = np.random.permutation(range(n_temp_x))[:flip2_n]
        perm3 = np.random.permutation(range(n_temp_x))[:rot_n]

        for n in xrange(flip1_n):
            temp_x[perm1[n]] = temp_x[perm1[n],:,::-1,:]

        for n in xrange(flip2_n):
            temp_x[perm2[n]] = temp_x[perm2[n],:,:,::-1]

        ############# ROTATIONS TAKEN OUT
        #for n in xrange(flip2_n):
        #    rand = np.random.randint(1,4)
        #    temp_x[perm2[n]] = np.rot90(temp_x[perm2[n]],rand)

        temp_x = temp_x.reshape(temp_x.shape[0],layers_3D*temp_x.shape[2]**2)

    train_set_x.set_value(temp_x.astype(np.float32), borrow = True)
    train_set_y.set_value(temp_y.astype(np.float32), borrow = True)

    return train_set_x,train_set_y

def dropout(X,p=0.5):
    '''
    Perform dropout with probability p
    '''
    if p>0:
        retain_prob = 1-p
        X *= srng.binomial(X.shape,p=retain_prob,dtype = theano.config.floatX)
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

def rectify(X): 
    '''
    Rectified linear activation function
    '''
    return T.maximum(X,0.)
    
    
# --------------------------------------------------------------------------
def init_predict(data_set,net,batch_size,x,index):
    predict = theano.function(inputs=[index], 
                            outputs=net.layer4.prediction(),
                            givens = {
                                x: data_set[index * batch_size: (index + 1) * batch_size]
                                }
                            )
    return predict

# --------------------------------------------------------------------------
def predict_set(predict, n_batches, classifier, pred_window_size, number_pixels = None):
                            
    if classifier == "membrane" or classifier == "synapse":
        output = np.zeros((0,pred_window_size[1]**2))
    elif classifier == "membrane_synapse":
        output = np.zeros((0,2*pred_window_size[1]**2))

    start_test_timer = time.time()
    for i in xrange(n_batches):
        output = np.vstack((output,predict(i)))
    stop_test_timer = time.time()

    if number_pixels != None:
        pixels_per_second = number_pixels/(stop_test_timer-start_test_timer)
        print "Prediction, pixels per second:  ", pixels_per_second

    return output

# --------------------------------------------------------------------------
def evaluate(pred,ground_truth,eval_window_size,classifier):
    
    ground_truth = ground_truth[:pred.shape[0]]

    #ground_truth = ground_truth[:pred.shape[0]]
    eval_window = np.ones((eval_window_size,eval_window_size))

    # Reshape
    if ground_truth.ndim == 2 or ground_truth.ndim == 3:
        if classifier in ["membrane","synapse"]:
            window_size = int(np.sqrt(ground_truth.shape[1]))
            ground_truth = ground_truth.reshape(ground_truth.shape[0],1,window_size,window_size)
        elif classifier == "membrane_synapse":
            window_size = int(np.sqrt(ground_truth.shape[1]/2))
            ground_truth = ground_truth.reshape(ground_truth.shape[0],2,window_size,window_size)
        pred = pred.reshape(ground_truth.shape)

        if window_size < eval_window_size:
            eval_window_size = window_size

    if classifier in ["membrane","synapse"]:
        # Calculate pixel-wise error
        pixel_error = np.mean(np.abs(pred-ground_truth))

        # Calculate window-wise error
        windowed_error = 0
        for n in xrange(pred.shape[0]):
            pred_conv         = signal.convolve2d(pred[n,0],eval_window,mode="valid")/float(eval_window_size**2)
            ground_truth_conv = signal.convolve2d(ground_truth[n,0],eval_window,mode="valid")/float(eval_window_size**2)
            pred_conv          = pred_conv[::eval_window_size,::eval_window_size]
            ground_truth_conv  = ground_truth_conv[::eval_window_size,::eval_window_size]

            windowed_error += np.mean(np.abs(pred_conv-ground_truth_conv))

        windowed_error = windowed_error/float(pred.shape[0])

    elif self.classifier=="membrane_synapse":
        # Calculate pixel-wise error
        pixel_error = []
        pixel_error.append(np.mean(np.abs(pred[:,0]-ground_truth[:,0])))
        pixel_error.append(np.mean(np.abs(pred[:,1]-ground_truth[:,1])))

        windowed_error = []
        for m in xrange(2):
            # Calculate window-wise error
            win_error = 0
            for n in xrange(pred.shape[0]):
                pred_conv         = signal.convolve2d(pred[n,m],eval_window,mode="valid")/float(eval_window_size**2)
                ground_truth_conv = signal.convolve2d(ground_truth[n,m],eval_window,mode="valid")/float(eval_window_size**2)
                pred_conv          = pred_conv[::eval_window_size,::eval_window_size]
                ground_truth_conv  = ground_truth_conv[::eval_window_size,::eval_window_size]

                win_error += np.mean(np.abs(pred_conv-ground_truth_conv))

            win_error = win_error/float(pred.shape[0])
            windowed_error.append(win_error)

    return pixel_error,windowed_error
