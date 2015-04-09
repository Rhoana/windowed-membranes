import theano
import theano.tensor as T
import numpy as np

class Functions(object):
    '''
    Class containing helper functions for the ConvNet class.
    '''

    @staticmethod
    def flip_rotate(X,Y,in_window_shape,out_window_shape):

        temp_x     = X.eval()
        temp_y     = Y.eval()

        temp_x     = temp_x.reshape(temp_x.shape[0],in_window_shape[0],in_window_shape[1])
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
            temp_x[perm1[n]] = temp_x[perm1[n],::-1,:]
            temp_y[perm1[n]] = temp_y[perm1[n],::-1,:]

        for n in xrange(flip2_n):
            temp_x[perm2[n]] = temp_x[perm2[n],:,::-1]
            temp_y[perm2[n]] = temp_y[perm2[n],:,::-1]

        for n in xrange(flip2_n):
            rand = np.random.randint(1,4)
            temp_x[perm2[n]] = np.rot90(temp_x[perm2[n]],rand)
            temp_y[perm2[n]] = np.rot90(temp_y[perm2[n]],rand)

        X,Y = theano.shared(temp_x),theano.shared(temp_y)
        return X,Y
    
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




