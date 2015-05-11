# Membrane and Synapse Detection
Convolutional network for membrane and synapse detection.

## How to run (default settings)
1. Place training data in `data`-folder. Data can be downloaded from [Google Drive](https://drive.google.com/drive/u/1/folders/0B016PpcCQHuVfmdYSEdxSGVHdDNuenJyQjdZdkRkUXVOamFzSEpua0hfSzNQX0xSLXpaMFU?ltmpl=drive).
2. Specify data-folders in `config/global.yaml`
3. Run default configuration: `python runner.py`
4. Plot latest run: `python plot.py`

## Config file
There are two main config files. Global settings are defined in `data/global.yaml` and custom settings is defined in a separate config-file. By default, this custom config-file is `data/default.yaml`. You can define your own custom config file `data/custom_config.yaml` and run it by `python runner.py custom_config.yaml`. Note that settings defined in the custom config-file will overide the settings in the global config-file.

### Settings

#### Network size

## Directory Structure
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

