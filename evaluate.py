from sklearn.metrics import f1_score, jaccard_similarity_score, roc_auc_score, classification_report, roc_curve, confusion_matrix
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import os
import seaborn as sns
from skimage.restoration import denoise_bilateral

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
        f1s[i] = f1_score(y_actual, y_thresh, labels=[0, 1], average='binary')
    return thresholds[np.argmax(f1s)]

def save_random_images(actual_pixel_labels, predicted_pixel_labels, num_images=3):
	indices = np.random.choice(actual_pixel_labels.shape[0], num_images, replace=False) 
	
	# create a samples folder within results if it does not yet exist
	if not os.path.exists("results/samples"):
		os.makedirs("results/samples")
	
	for i, idx in enumerate(indices):
	    #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
	    #f.set_size_inches(17, 5)
	    #ax1.imshow(np.around(actual_pixel_labels[idx]), cmap='bone',  interpolation='nearest')
	    #ax2.imshow(np.around(predicted_pixel_labels[idx]), cmap='bone',  interpolation='nearest')
	    #ax1.set_title("Actual synapses", fontsize="xx-large")
	    #ax2.set_title("Predicted synapses", fontsize="xx-large")
	    #ax1.set_axis_off()
	    #ax2.set_axis_off()
	    #f.savefig("results/samples/comparison_" + str(idx) + ".png")
	    misc.imsave('results/samples/predicted_' + str(i) + '.png', np.around(predicted_pixel_labels[idx]))
	    misc.imsave('results/samples/actual_' + str(i) + '.png', np.around(actual_pixel_labels[idx]))
	return True

def output_stats(actual_pixel_labels, predicted_pixel_labels):
	y_actual = np.around(actual_pixel_labels.flatten())    
	p_thresh = get_best_threshold_by_f1(y_actual, predicted_pixel_labels.flatten())
	y_predicted = threshold_array(predicted_pixel_labels.flatten(), p_thresh)

	f1 = f1_score(y_actual, y_predicted, labels=[0, 1], average='binary')
	jaccard = jaccard_similarity_score(y_actual, y_predicted)
	auc = roc_auc_score(y_actual, y_predicted)
	report = classification_report(y_actual, y_predicted, target_names=["non-synapses", "synapses"])
	confusion_mat = confusion_matrix(y_actual, y_predicted)

	output_str  = "Probability threshold is " + str(p_thresh)
	actual = np.around(actual_pixel_labels.flatten())    
	output_str +=  "F1 score is " + str(f1) 
	output_str += "\nJaccard similarity score is " + str(jaccard)
	output_str += "\nROC AUC score is " + str(auc)
	output_str += "\nFull Classification Report:\n " + str(report)

	print output_str
	with open("results/output_stats.txt", "w") as text_file:
	    text_file.write(output_str)

def save_roc_curve(actual_pixel_labels, predicted_pixel_labels):
	y_actual = actual_pixel_labels.flatten()
	y_predicted = predicted_pixel_labels.flatten()
	auc = roc_auc_score(y_actual, y_predicted)
	fpr, tpr, thresholds = roc_curve(y_actual, y_predicted)
	plt.figure(figsize=(12, 7))
	plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate', fontsize="xx-large")
	plt.ylabel('True Positive Rate', fontsize="xx-large")
	plt.title('Receiver Operating Characteristic', fontsize="xx-large")
	plt.legend(loc="lower right")
	plt.savefig("results/roc_curve.png")
	plt.show()
	return True

BILATERAL_FILTERING = True
actual_pixel_labels, predicted_pixel_labels = np.load("results/y.npy"), np.load("results/output.npy") 
if BILATERAL_FILTERING:
	predicted_pixel_labels = np.array([denoise_bilateral(im) for im in predicted_pixel_labels])
save_random_images(actual_pixel_labels, predicted_pixel_labels, 3)
output_stats(actual_pixel_labels, predicted_pixel_labels)
#save_roc_curve(actual_pixel_labels, predicted_pixel_labels)



