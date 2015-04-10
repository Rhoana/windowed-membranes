# Directory Structure


README.md
├── edge_prediction
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── read_img.py
│   │   ├── train-input
│   │   ├── train-labels
│   │   ├── x_test.npy
│   │   ├── x_train.npy
│   │   ├── y_test.npy
│   │   └── y_train.npy
│   ├── edge_prediction.py
│   ├── edge_prediction_conv
│   │   ├── __init__.py
│   │   ├── edge_cov_net.py
│   │   ├── helper_functions.py
│   ├── plot.py
│   ├── results
│   │   ├── output.npy
│   │   ├── x.npy
│   │   └── y.npy
│   └── util
│       ├── __init__.py
│       ├── pre_process.py
        |── functions.py
├── lib
│   ├── __init__.py
│   ├── hidden_layer.py
│   ├── logistic_sgd.py
│   ├── pool_layer.py
└── util

# Edge Prediction
Convolutional network that performs edge detection by using segmented labeled training data. 

There are two ways to run the code from within the edge_prediction directory:

## Option 1 (Run)
1 . Run `python edge_prediction --small/medium/large` to only perform edge prediction.
2. The training time is set to 100 epochs. To train for a shorter period, press `Ctrl+C` to throw a KeybordInterrupt and the program will exit the training loop and start the prediction on the test set. 
3. Run `plot.py`n to plot a visual prediction from the test set, where the integer n is
   a member of the test set.

## Option 2 (Run Model + Generate Training Data)
1. Place training data in `synapse_train_data/train_input`
2. Place training labels in `synapse_train_data/train_labels`
3. Run `python edge_prediction --pre-process --small/medium/large` to generate training/test data, and to predict edges.
4. Or run `python edge_prediction --pre-process only` to only generate training data
5. The training time is set to 100 epochs. To train for a shorter period, press `Ctrl+C` to throw a KeybordInterrupt and the program will exit the training loop and start the prediction on the test set. 
6. Run `plot.py`n to plot a visual prediction from the test set, where the integer n is
   a member of the test set.

Small, medium and large are convolutional network with 10,64 and 100 filters per
convolution. The train/test set for the three options are 1500/500, 4000/1000
and 9000/1000, respectively. In all cases, the validation set is of size 200
and is a subset of the test set. The number of neurons in the fully connected layer is always the same as the number of outputs in the last convolutional layer.