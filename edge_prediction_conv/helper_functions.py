import theano
import theano.tensor as T
import numpy as np

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
            #for n in xrange(flip2_n):
            #    rand = np.random.randint(1,4)
            #    temp_x[perm2[n]] = np.rot90(temp_x[perm2[n]],rand)
            #    temp_y[perm2[n]] = np.rot90(temp_y[perm2[n]],rand)

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

        train_set_x.set_value(temp_x, borrow = True)
        train_set_y.set_value(temp_y, borrow = True)

        return train_set_x,train_set_y

    @staticmethod
    def make_random_matrix(size,poolsize):
        threshold = 100
        
        fan_in = np.prod(size[1:])                                   
        fan_out = (size[0] * np.prod(size[2:]) /
        np.prod(poolsize))                                        
                                                            
        # Initialize weight matrix                                              
        W_bound = np.sqrt(6. / (fan_in + fan_out))    

        max_cond = np.infty                                                  
        count = 1

        random_matrix = np.zeros(size)

        for n in xrange(size[0]):
            while max_cond > threshold:                                             
                temp = np.random.uniform(low=-W_bound, high=W_bound, size=size[1:])
                U, s, V = np.linalg.svd(temp, full_matrices=True)       
                                                                    
                max_cond = np.max(np.max(s,axis=1)/np.min(s,axis=1))       
                count += 1

            random_matrix[n] = temp
                                                                    
        print 'Maximum condition number: ',max_cond, count
        return random_matrix
    
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




