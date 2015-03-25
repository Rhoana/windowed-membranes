# Edge Classifier
The edge classifer can be run with a small predefined training-set or
a training set with real and labeled segmented images.

## Option 1

1. Run ConvNet.py with small predefined training/test set

## Option 2

1. Place training data in *"synapse_train_data/train_input"*
2. Place training labels in *"synapse_train_data/train_labels"*
3. Run `python edge_prediction --pre-process` to generate training/test data, and to predict edges
4. Or run `python edge_prediction` to only perform edge prediction.




