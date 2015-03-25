# Edge Classifier
Convolutional network that performs edge detection by using segmented labeled training data. 

## Run
1. Place training data in `synapse_train_data/train_input`
2. Place training labels in `synapse_train_data/train_labels`
3. Run `python edge_prediction --pre-process --small/medium/large` to generate training/test data, and to predict edges
4. Or run `python edge_prediction --small/medium/large` to only perform edge prediction.
5. The training time is set to 100 epochs. To train for a shorter period, press `Ctrl+C` to throw a KeybordInterrupt and the program will exit the training
   loop and start the prediction on the test set. 
6. Run `plot.py`n to plot a visual prediction from the test set, where n(int) is
   a member of the test set.

Small, medium and large are convolutional network with 10,32 and 64 filters per
convolution. The train/test set for the three options are 1500/500, 4000/1000
and 9000/1000, respectively. In all cases, the validation set is of size 200
and is a subset of the test set. The number of neurons in the fully connected layer is always the same as the number of outputs in the last convolutional layer.