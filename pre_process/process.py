from skimage import exposure
import numpy as np
class Process(object):

    def process_images(self,img_input,train_img_labels,classifier,membrane_edges):
        """
        Process images
        
        Output:
        - img_input
        - labeled_in - labeled data used for sampling of training set
        - labeled_out - labeled data used as ground truth during training 
        """

        labeled_in = train_img_labels

        if classifier == 'membrane':
            if membrane_edges == 'WideEdges':
                labeled_out = np.zeros(labeled_in.shape)
                labeled_out[:,0] = self.thick_edge(labeled_in[:,0])
            elif membrane_edges == 'GaussianBlur':
                labeled_out = self.membrane_gaussian(train_img_labels)
            else:
                labeled_out = labeled_in
                print "Warning: thin edge"

        elif classifier == 'synapse':
            labeled_out = labeled_in

        elif classifier == "membrane_synapse":
            if membrane_edges == "WideEdges":
                labeled_out = np.zeros(labeled_in.shape)
                labeled_out[:,0] = self.thick_edge(labeled_in[:,0])
                labeled_out[:,1] = labeled_in[:,1]
            else:
                print "Warning: thin edge"

        return img_input,labeled_in,labeled_out

    def membrane_gaussian(self,labeled_in,sigma=3):
        """
        Gaussian smoothed edges
        """
        for n in range(train_img_labels.shape[0]):
            labeled_out[n] = self.thick_edge(labeled_in[n])
        return labeled_out


    def eqh(self,train_img_input):
        for n in range(train_img_input.shape[0]):
            train_img_input[n] = exposure.equalize_adapthist(train_img_input[n], clip_limit=0.03)
        return train_img_input

    def thick_edge(self,labeled_in):
        labeled_out = np.zeros(np.shape(labeled_in))
        for n in range(labeled_in.shape[0]):
            labeled_out[n] = self.thick_edge_one(labeled_in[n])
        return labeled_out

    def thick_edge_one(self,img):
        """
        Thick labeled edge
        """
        thick = np.zeros(np.shape(img))

        for n in xrange(1,img.shape[0]-1):
            for m in xrange(1,img.shape[1]-1):
                if img[n,m] == 1:
                    thick[n,m] = 1
                    thick[n-1,m] = 1
                    thick[n-1,m] = 1
                    thick[n,m+1] = 1
                    thick[n,m-1] = 1
                    thick[n-1,m-1] = 1
                    thick[n-1,m+1] = 1
                    thick[n+1,m+1] = 1
                    thick[n+1,m-1] = 1
        return thick
