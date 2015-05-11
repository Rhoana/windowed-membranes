# Membrane and Synapse Detection
Convolutional network for membrane and synapse detection.

### Dependencies
* Numpy
* Theano
* Matplotlib (for plotting)
* YAML
* PIL
* Scipy

### Install
Install and upgrade all dependencies:

`pip install -U numpy theano matplotlib pyyaml pil scipy `

Clone repository:

`git clone https://github.com/Rhoana/windowed-prediction.git`

### How to run (default settings)
1. Place training data in `data`-folder. Data can be downloaded from [Google Drive](https://drive.google.com/drive/u/1/folders/0B016PpcCQHuVfmdYSEdxSGVHdDNuenJyQjdZdkRkUXVOamFzSEpua0hfSzNQX0xSLXpaMFU?ltmpl=drive).
2. Specify data-folders in `config/global.yaml`
3. Run default configuration: `python runner.py`
4. Plot latest run: `python plot.py`

### Config file
There are two main config files. Global settings are defined in `data/global.yaml` and custom settings is defined in a separate config-file. By default, this custom config-file is `data/default.yaml`. You can define your own custom config file `data/custom_config.yaml` and run it by `python runner.py custom_config.yaml`. Note that settings defined in the custom config-file will overide the settings in the global config-file.

### Documentation

Coming...

### Contact

Hallvard Moian Nydal - hallvardnydal@gmail.com



