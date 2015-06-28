import numpy as np
import cPickle
import mahotas as mh
import partition_comparison
import matplotlib.pyplot as plt

cmap = matplotlib.colors.ListedColormap ( np.random.rand ( 256,3))

def remove_synapse(img, threshold = 399):  
    for m in xrange(img.shape[0]):
        for n in xrange(img.shape[1]):
            if img[m,n] > threshold:
                img[m,n] = 0
    return img

def watershed(probs, radius):
    if probs.ndim == 3:
        shed = np.zeros(probs.shape)

        for n in xrange(probs.shape[0]):
            sel = disk(radius)
            minima = mh.regmin(probs[n], Bc=sel)
            markers, nr_markers = mh.label(minima)
            shed[n] = mh.cwatershed(probs[n], markers)

    else:
        sel = disk(radius)
        minima = mh.regmin(probs, Bc=sel)
        markers, nr_markers = mh.label(minima)
        shed = mh.cwatershed(probs, markers)

    return shed

def disk(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return x**2 + y**2 <= radius**2

def partition_VI(putative, gold):
    mask = gold != 0
    VI_res = partition_comparison.variation_of_information(gold[mask], putative[mask].astype(gold.dtype))
    return VI_res

address1          = "results/48x48_ed/"
pred_48x48         = np.load(address1 + "/results/output.npy")
y1                = np.load(address1 + "/results/y.npy")
pred_window_size  = np.load(address1 + "/results/pred_window_size.npy")
gap = (pred_window_size[0]-pred_window_size[1])/2

# Load in test-addresses
f = file(address1 + "/pre_processed/" + "test_adress.dat", 'rb')
test_address = cPickle.load(f)
f.close()

pred_48x48 = pred_48x48[:,0]

ground_truth = np.zeros(pred_48x48.shape)
n = 0
for address_img in test_address:
    ground_truth_new = mh.imread(address_img)
    ground_truth[n] = ground_truth_new[gap:-gap,gap:-gap]
    ground_truth[n] = ground_truth[n].astype(np.uint16)
    ground_truth[n] = remove_synapse(ground_truth[n])
    n +=1

#plt.figure()
#plt.imshow(img)
#plt.figure()
#plt.imshow(ground_truth)
#plt.show()
#exit()

#plt.figure()
#plt.imshow(img)
#plt.figure()
#plt.imshow(ground_truth,cmap=plt.cm.prism)
#plt.figure()
#plt.imshow(watershed(img,5),cmap=plt.cm.prism)

from gala import imio, classify, features, agglo, evaluate as ev

prob_map           = pred_48x48
wshed              = watershed(prob_map,5)

ground_truth_train = ground_truth[:9]
prob_map_train     = prob_map[:9]
watershed_train    = wshed[:9]

ground_truth_test  = ground_truth[-1]
prob_map_test      = prob_map[-1]
watershed_test     = wshed[-1]

# create a feature manager
fm = features.moments.Manager()
fh = features.histogram.Manager()
fc = features.base.Composite(children=[fm,fh])

print "Starting agglo.."

g_train = agglo.Rag(watershed_train, prob_map_train,feature_manager=fc)

print "Making train set"

(X,y,w,merges) = g_train.learn_agglomerate(ground_truth_train,fc)[0]
y = y[:,0]

print "Training classifier"

rf = classify.DefaultRandomForest().fit(X, y)
learned_policy = agglo.classifier_probability(fc, rf)

print "Predicting"

g_test = agglo.Rag(watershed_test,prob_map_test,learned_policy,feature_manager=fc)
g_test.agglomerate(0.5)
seg_test1 = g_test.get_segmentation()

print "VI gala:",partition_VI(seg_test1, ground_truth_test) 

radii = np.linspace(7, 85, 15)
min_VI = 99.
for r in radii:
    VI = partition_VI(watershed(prob_map_test,r), ground_truth_test)
    if VI<min_VI:
        min_VI = VI

print "VI, watershed:",min_VI

plt.figure()
plt.imshow(ground_truth_train,cmap=cmap)
plt.figure()
plt.imshow(seg_test1,cmap=cmap)
plt.show()



