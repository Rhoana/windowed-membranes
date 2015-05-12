from skimage import exposure
import numpy as np
class Process(object):

    def process_images(self,train_img_input,train_img_labels,classifier,membrane_edges,adaptive_histogram_equalization):

        labeled_in = train_img_labels

        if classifier == 'membrane':
            if membrane_edges == 'WideEdges':
                labeled_out = self.thick_edge(labeled_in)
            elif membrane_edges == 'GaussianBlur':
                labeled_out = self.membrane_gaussian(train_img_labels)
            else:
                labeled_out = labeled_in
                print "Warning: thin edge"

        elif classifier == 'synapse':
            labeled_out = labeled_in

        #if adaptive_histogram_equalization:
        #    train_img_input = self.eqh(train_img_input)

        return train_img_input,labeled_in,labeled_out

    def membrane_gaussian(self,labeled_in,sigma=3):
        for n in range(train_img_labels.shape[0]):
            labeled_out[n] = self.thick_edge(labeled_in[n])
        return labeled_out

    def membrane_thickedge(self,labeled_in,sigma=3):
        for n in range(train_img_labels.shape[0]):
            labeled_out[n] = scipy.ndimage.gaussian_filter(labeled_in[n], sigma=sigma)
            labeled_out[n] = labeled_out[n]/labeled_out[n].max()
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

    def thick_edge_one(self,imarray):
        thickarray = np.zeros(np.shape(imarray))

        for n in xrange(1,imarray.shape[0]-1):
            for m in xrange(1,imarray.shape[1]-1):
                if imarray[n,m] == 1:
                    thickarray[n,m] = 1
                    thickarray[n-1,m] = 1
                    thickarray[n-1,m] = 1
                    thickarray[n,m+1] = 1
                    thickarray[n,m-1] = 1
                    thickarray[n-1,m-1] = 1
                    thickarray[n-1,m+1] = 1
                    thickarray[n+1,m+1] = 1
                    thickarray[n+1,m-1] = 1
        return thickarray

        return train_img_input,labeled_in,labeled_out
