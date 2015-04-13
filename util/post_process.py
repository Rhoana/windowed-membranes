import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import sys

def post_process():

    img_shape = (1024,1024)                                                         
    in_window_shape = (64,64)                                                       
    out_window_shape = (48,48)                                                      
    diff = in_window_shape[0]-out_window_shape[0]    

    output = np.load('results/output.npy')
    y      = np.load('results/y.npy')                                              
    table  = np.load('data/table.npy')

    y      = y.reshape(y.shape[0],out_window_shape[0],out_window_shape[1])  
    output = output.reshape(output.shape[0],out_window_shape[0],out_window_shape[1])         
          
    nr_images    = np.max(table[:,0]) + 1 
    y_whole      = np.zeros((nr_images, img_shape[0]-diff, img_shape[0]-diff))
    output_whole = np.zeros((nr_images, img_shape[0]-diff, img_shape[0]-diff))
    count        = np.zeros((nr_images, img_shape[0]-diff, img_shape[0]-diff))

    for i in xrange(table.shape[0]):       
        y_whole[table[i,0],(table[i,1]):(table[i,1]+out_window_shape[0]),(table[i,2]):(table[i,2] + out_window_shape[0])]      += y[i]      
        output_whole[table[i,0],(table[i,1]):(table[i,1]+out_window_shape[0]),(table[i,2]):(table[i,2] + out_window_shape[0])] +=  output[i] 
        count[table[i,0],(table[i,1]):(table[i,1]+out_window_shape[0]),(table[i,2]):(table[i,2] + out_window_shape[0])] += np.ones((out_window_shape[0],out_window_shape[1]))                                              

    count = count.astype(np.float32)
    
    y_whole      /= count
    output_whole /= count
    
    np.save('results/y_whole.npy',y_whole)
    np.save('results/output_whole.npy',output_whole)


    
if __name__ == "__main__":
    post_process()
