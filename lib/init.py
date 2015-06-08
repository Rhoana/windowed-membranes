import numpy as np

from util.utils import floatX

def glorotNormal(shape, gain=np.sqrt(2), mean = 0.0):

    n1, n2 = shape[:2]
    receptive_field_size = np.prod(shape[2:])

    std = gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
    
    return floatX(np.random.normal(mean, std, size=shape))
    
def glorotUniform(shape, gain=np.sqrt(2), mean = 0.0):

    n1, n2 = shape[:2]
    receptive_field_size = np.prod(shape[2:])

    std = gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
    a = mean - np.sqrt(3) * std
    b = mean + np.sqrt(3) * std
    
    return floatX(np.random.uniform(low=a, high=b, size=shape))


def HeNormal(shape, gain= np.sqrt(2), mean = 0.0):
    fan_in = np.prod(shape[1:])

    std = gain * np.sqrt(1.0 / fan_in)
    return floatX(np.random.normal(mean, std, size=shape))


def HeUniform(shape, gain= np.sqrt(2)):
    fan_in = np.prod(shape[1:])

    std = self.gain * np.sqrt(1.0 / fan_in)
    a = mean - np.sqrt(3) * std
    b = mean + np.sqrt(3) * std
    
    return floatX(np.random.uniform(low=a, high=b, size=shape))
    
def constant(shape, val):
    return floatX(np.ones(shape) * val)
