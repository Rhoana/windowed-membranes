import mahotas as mh
import cPickle
from gala import evaluate as ev
import numpy as np

def disk(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return x**2 + y**2 <= radius**2


def watershed(probs, radius):
    sel = disk(radius)
    minima = mh.regmin(probs, Bc=sel)
    markers, nr_markers = mh.label(minima)
    return mh.cwatershed(probs, markers)

def VI(putative, gold):
    mask = gold != 0
    VI_res = partition_comparison.variation_of_information(gold[mask], putative[mask].astype(gold.dtype))
    return VI_res

def evaluate_VI(adress):
    
    pred = np.load(adress + "/results/output.npy")[:,0]
    pred_window_size = np.load(adress + "/results/pred_window_size.npy")
        
    # Load in test-addresses
    f = file(adress + "/pre_processed/" + "test_adress.dat", 'rb')
    test_adress = cPickle.load(f)
    f.close()

    gap = (pred_window_size[0]-pred_window_size[1])/2
    radii = np.linspace(7, 65, 10) 
    
    VI_metric = np.zeros((radii.size,3))
    n = 0
    for adress_img in test_adress:
        ground_truth = mh.imread(adress_img)
        ground_truth = ground_truth[gap:-gap,gap:-gap]
        ground_truth = ground_truth.astype(np.uint16)
        
        r = 0
        for radius in radii:
            # VI_metric[0] = undersegmentation error
            # VI_metric[1] = oversegmentation error
            VI_split =  ev.split_vi(watershed(pred[n], radius), ground_truth)
            VI_metric[r,0] += VI_split[0]
            VI_metric[r,1] += VI_split[1]
            VI_metric[r,2] += VI_split[0] + VI_split[1]
            r += 1
            
        n += 1        
    
    VI_metric /= n 
    
    VI_min_pos = np.argmin(VI_metric[:,2])
    VI_min     = VI_metric[VI_min_pos]

    print "Variation of Information (VI):",VI_min[2]
    print "VI, undersegmentation error:", VI_min[0]
    print "VI, oversegmentation error:", VI_min[1]
    
    np.save(adress + "/results/VI.npy",VI_metric)
    
    
    return VI_min

if __name__ == "__main__":

    latest_res = open('latest_run.txt','r')
    adress = latest_res.readlines()
    latest_res.close()
    adress = adress[0].split('\n')[0]
    adress = str(adress)

    evaluate_VI(adress)
