# Directory Structure

<pre>
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
</pre>

# Membrane and Synapse Detection
Convolutional network for membrane and synapse detection.

## How to run (default settings)
1. Place training data in `data`-folder
2. If training-data is in tiff-stack format, run: `python data/clean_folders.py`
3. Specify data-folders in `config/global.yaml`
4. Run default configuration: `python runner.py`
5. Plot latest run: `python plot.py`

## Config file
There are two main config files. Global settings are defined in `data/global.yaml`. These settings are overridden by the settings defined in your custom config-file. Running `python runner.py` will run the default custom config-file. To use a customized config-file you can define a file `data/custom_config.yaml` and run it by `python runner.py custom_config.yaml`

### Settings

#### Network size

