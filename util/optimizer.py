import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict  

class Optimizer(object):

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




