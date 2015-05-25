import theano
import theano.tensor as T
import numpy as np

from util.utils import *
from collections import OrderedDict

class Functions(object):
    '''
    Class containing helper functions for the ConvNet class.
    '''

    @staticmethod
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
        
        
    def init_optimizer(self, optimizer, cost, params, optimizerData):
        '''
        Choose between different optimizers 
        '''
        if optimizer == 'stochasticGradient':
            try:
                updates = self.stochasticGradient(cost, 
                                              params,
                                              lr      = optimizerData['learning_rate'])
            except:
                updates = self.stochasticGradient(cost, 
                                              params)
                print "Warning: Using default optimizer settings."
                
        elif optimizer == 'rmsprop':    
            try:
                updates = self.rmsprop(cost, params, 
                                                 lr      = optimizerData['learning_rate'],
                                                 rho     = optimizerData['rho'],
                                                 epsilon = optimizerData['epsilon'])
            except:
                updates = self.rmsprop(cost, params)
                print "Warning: Using default optimizer settings."
        
        elif optimizer == 'Adam':    
            try:
                updates = self.Adam(cost, params, 
                                                 lr      = optimizerData['learning_rate'],
                                                 b1      = optimizerData['b1'],
                                                 b2      = optimizerData['b2'],
                                                 e       = optimizerData['e'])
            except:
                updates = self.Adam(cost, params)
                print "Warning: Using default optimizer settings."
        
        elif optimizer == 'adadelta':    
            try:
                updates = self.adadelta(cost, params, 
                                                 lr      = optimizerData['learning_rate'],
                                                 rho     = optimizerData['rho'],
                                                 epsilon = optimizerData['epsilon'])
            except:
                updates = self.adadelta(cost, params)
                print "Warning: Using default optimizer settings."
        elif optimizer == 'adagrad':    
            try:
                updates = self.adagrad(cost, params, 
                                                 lr      = optimizerData['learning_rate'],
                                                 epsilon = optimizerData['epsilon'])
            except:
                updates = self.adagrad(cost, params)
                print "Warning: Using default optimizer settings."
                
        elif optimizer == 'momentum':    
            try:
                updates = self.momentum(cost, params, 
                                                 lr      = optimizerData['learning_rate'],
                                                 momentum = optimizerData['momentum'])
            except:
                updates = self.momentum(cost, params)
                print "Warning: Using default optimizer settings."
                          
        return updates

    def rmsprop(self,cost, params, lr = 0.01, rho=0.9, epsilon=1e-6):
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
        
    def stochasticGradient(self,cost,params,lr = 0.1):
        '''
        Stochastic Gradient Descent
        '''
        grads = T.grad(cost, params)
        updates = OrderedDict()

        for param, grad in zip(params, grads):
            updates[param] = param - lr * grad

        return updates
        
    def Adam(self,cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        """
        lasagne
        """
        updates = []
        grads = T.grad(cost, params)
        i = theano.shared(floatX(0.))
        i_t = i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
         m = theano.shared(p.get_value() * 0.)
         v = theano.shared(p.get_value() * 0.)
         m_t = (b1 * g) + ((1. - b1) * m)
         v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
         g_t = m_t / (T.sqrt(v_t) + e)
         p_t = p - (lr_t * g_t)
         updates.append((m, m_t))
         updates.append((v, v_t))
         updates.append((p, p_t))
        updates.append((i, i_t))
        return updates    
    
    def adadelta(self,cost, params, learning_rate=0.01, rho=0.95, epsilon=1e-6):

        grads = T.grad(cost,params)
        updates = OrderedDict()

        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            # accu: accumulate gradient magnitudes
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            # delta_accu: accumulate update magnitudes (recursively!)
            delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                       broadcastable=param.broadcastable)

            # update accu (as in rmsprop)
            accu_new = rho * accu + (1 - rho) * grad ** 2
            updates[accu] = accu_new

            # compute parameter update, using the 'old' delta_accu
            update = (grad * T.sqrt(delta_accu + epsilon) /
                      T.sqrt(accu_new + epsilon))
            updates[param] = param - learning_rate * update

            # update delta_accu (as accu, but accumulating updates)
            delta_accu_new = rho * delta_accu + (1 - rho) * update ** 2
            updates[delta_accu] = delta_accu_new

        return updates  
        
    def adagrad(self, cost, params, learning_rate=0.01, epsilon=1e-6):
        """
        lasagne
        """

        grads = T.grad(cost, params)
        updates = OrderedDict()

        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            accu_new = accu + grad ** 2
            updates[accu] = accu_new
            updates[param] = param - (learning_rate * grad /
                                      T.sqrt(accu_new + epsilon))

        return updates
        

    def apply_momentum(self, updates, params=None, momentum=0.9):
        """
        lasagne
        """

        if params is None:
            params = updates.keys()
        updates = OrderedDict(updates)

        for param in params:
            value = param.get_value(borrow=True)
            velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                     broadcastable=param.broadcastable)
            x = momentum * velocity + updates[param]
            updates[velocity] = x - param
            updates[param] = x

        return updates


    def momentum(self, cost, params, lr = 0.1, momentum=0.9):
        """
        lasagne
        """
        updates = self.stochasticGradient(cost, params, lr = lr)
        return self.apply_momentum(updates, momentum=momentum)




