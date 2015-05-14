import theano
import theano.tensor as T
import numpy as np

from numpy.random import normal
from scipy.signal import convolve2d
from scipy import interpolate

float_x = theano.config.floatX

class InLayer(object):
    
    def __init__(self,batch_size, in_window_shape, out_window_shape, pred_window_size, layers_3D):
        
        self.batch_size       = batch_size
        self.in_window_shape  = in_window_shape
        self.out_window_shape = out_window_shape
        self.pred_window_size = pred_window_size
        self.layers_3D        = layers_3D
        self.in_diff          = (in_window_shape[0] - pred_window_size[0])/2
        self.out_diff         = (out_window_shape[0] - pred_window_size[1])/2

    def in_layer(self, input, input_labeled, alpha=0,angle = 0, p_noise = 0): #alpha=10,angle=10
        """
        elastic transform
        """
        
        input = input.reshape((self.batch_size,self.layers_3D,self.in_window_shape[0],self.in_window_shape[1]))
        input_labeled = input_labeled.reshape((self.batch_size,self.out_window_shape[0],self.out_window_shape[1]))
    
        srs = T.shared_randomstreams.RandomStreams()
        target = T.as_tensor_variable(np.indices((self.in_window_shape[0],self.in_window_shape[0])))
        
        
        if alpha or angle:
            if alpha:
                # create random displacement fields
                dis = self.in_window_shape[0]
                displacement_field_x = np.array([[normal(x,alpha) for x in [0,dis]] \
                                        for y in [0,dis]])
                displacement_field_y = np.array([[normal(y,alpha) for x in [0,dis]] \
                                        for y in [0,dis]])
                                        
                displacement_field_x -= displacement_field_x.mean()
                displacement_field_y -= displacement_field_y.mean()

                xy = np.array([0,self.in_window_shape[0]])
                f_x = interpolate.interp2d(xy,xy,displacement_field_x, kind='linear')
                f_y = interpolate.interp2d(xy,xy,displacement_field_y, kind='linear')
                xy = np.arange(self.in_window_shape[0])

                target = T.as_tensor_variable(np.array([f_x(xy, xy),f_y(xy, xy)]))

    
            if angle:
                theta = angle * np.pi / 180 * srs.uniform(low=-1)
                c, s = T.cos(theta), T.sin(theta)
                rotate = T.stack(c, -s, s, c).reshape((2,2))
                target = T.tensordot(rotate, target, axes=((0, 0)))
    

            # Clip the mapping to valid range and linearly interpolate
            if alpha or angle:
                transx = T.clip(target[0], 0, self.in_window_shape[0]  - 1 - .001)
                transy = T.clip(target[1], 0, self.in_window_shape[1]  - 1 - .001)

                topp = T.cast(transy, 'int32')
                left = T.cast(transx, 'int32')
                fraction_y = T.cast(transy - topp, float_x)
                fraction_x = T.cast(transx - left, float_x)

                output = input[:,:, topp, left] * (1 - fraction_y) * (1 - fraction_x) + \
                         input[:,:, topp, left + 1] * (1 - fraction_y) * fraction_x + \
                         input[:,:, topp + 1, left] * fraction_y * (1 - fraction_x) + \
                         input[:,:, topp + 1, left + 1] * fraction_y * fraction_x
                output_labeled = input_labeled[:,topp, left] * (1 - fraction_y) * (1 - fraction_x) + \
                                 input_labeled[:,topp, left + 1] * (1 - fraction_y) * fraction_x + \
                                 input_labeled[:,topp + 1, left] * fraction_y * (1 - fraction_x) + \
                                 input_labeled[:,topp + 1, left + 1] * fraction_y * fraction_x
        else:
            output = input
            output_labeled = input_labeled
            
        #Now add some noise
        if p_noise:
            noise = srs.normal((self.batch_size,self.layers_3D,self.in_window_shape[0],self.in_window_shape[0]))*p_noise
            output += noise
            
        if self.in_diff != 0:
            output         = output[:,:,self.in_diff:(-self.in_diff),self.in_diff:(-self.in_diff)]
        else:
            output = output
        if self.out_diff != 0:
            output_labeled = output_labeled[:,self.out_diff:(-self.out_diff),self.out_diff:(-self.out_diff)]
        else:
            output_labeled = output_labeled
        
        self.output = output.reshape((self.batch_size,self.layers_3D,self.pred_window_size[0],self.pred_window_size[0]))
        self.output_labeled = output_labeled.reshape((self.batch_size,self.pred_window_size[1]**2))
                       
    
if __name__ == "__main__":
    pass