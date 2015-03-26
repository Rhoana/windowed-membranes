import numpy as np
import theano
import theano.tensor as T

class Functions(object):
    '''
    Class containing helper functions for the ConvNet class.
    '''
    
    def stack(self,stackitems):
        '''
        Works like vstack for theano tensors 
        '''
        for n in xrange(len(stackitems)-1):
            if n == 0:
                output = T.concatenate((stackitems[n],stackitems[n+1]),axis=1)
            else:
                output = T.concatenate((output,stackitems[n+1]),axis=1)
        return output
    
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