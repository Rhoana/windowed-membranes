import numpy as np
from sklearn.metrics import f1_score, jaccard_similarity_score, roc_auc_score, classification_report, roc_curve, confusion_matrix
from scipy import signal

def threshold_array(array, threshold):                                          
    y_threshold = np.copy(array)                                                
    y_threshold[np.where( array > threshold )] = 1                              
    y_threshold[np.where( array <= threshold )] = 0                             
    return y_threshold                                                          
            
def get_best_threshold_by_f1(y_actual, y_pred):                                 
    f1s = np.empty((1000))                                                      
    thresholds = np.linspace(0.01, 1, 1000)                                     
    for i, threshold in enumerate(thresholds):                                  
        y_thresh = threshold_array(y_pred, threshold)                           
        f1s[i] = f1_score(y_actual, y_thresh, labels=[0, 1])                    
    return thresholds[np.argmax(f1s)]      

def output_stats(actual_pixel_labels, predicted_pixel_labels):                  
    y_actual = np.around(actual_pixel_labels.flatten())                         
    p_thresh = get_best_threshold_by_f1(y_actual, predicted_pixel_labels.flatten())
    y_predicted = threshold_array(predicted_pixel_labels.flatten(), p_thresh)   
                                                                        
    f1 = f1_score(y_actual, y_predicted, labels=[0, 1])                         
    jaccard = jaccard_similarity_score(y_actual, y_predicted)                   
    auc = roc_auc_score(y_actual, y_predicted)                                  
    report = classification_report(y_actual, y_predicted, target_names=["non-synapses", "synapses"])
    confusion_mat = confusion_matrix(y_actual, y_predicted)                     
                                                                            
    output_str  = "Probability threshold is " + str(p_thresh) + "\n"                   
    actual = np.around(actual_pixel_labels.flatten())                           
    output_str +=  "F1 score is " + str(f1)                                     
    output_str += "\nJaccard similarity score is " + str(jaccard)               
    output_str += "\nROC AUC score is " + str(auc)                              
    output_str += "\nFull Classification Report:\n " + str(report)  

    print output_str 

if __name__ == "__main__":
    latest_res = open('latest_run.txt','r')                                     
    adress = latest_res.readlines()                                             
    latest_res.close()                                                          
    adress = adress[0].split('\n')[0]                                           
    adress = str(adress)   
    pred = np.load(adress + "/results/output.npy")
    y    = np.load(adress + "/results/y.npy")

    if pred.shape[1] == 2:
        pred = pred[:,1]
        y = y[:,1]
    else:
        pred = pred[:,0]
        y = y[:,0]

    print "-"*50
    print "Pixel-wise"

    output_stats(y,pred)

    eval_window_size = 6
    eval_window = np.ones((eval_window_size,eval_window_size)) 
    for n in xrange(pred.shape[0]):                                 
        pred_conv         = signal.convolve2d(pred[n],eval_window,mode="valid")/float(eval_window_size**2)
        ground_truth_conv = signal.convolve2d(y[n],eval_window,mode="valid")/float(eval_window_size**2)
        pred_conv          = pred_conv[::eval_window_size,::eval_window_size]
        ground_truth_conv  = y_conv[::eval_window_size,::eval_window_size]

    print "-"*50
    print "Window-wise"
    print "Evaluation window:",eval_window_size
    output_stats(y_conv,pred_conv)



