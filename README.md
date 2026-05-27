### ac_segmentation

This repository contains code for segmenting data from light microscopy images.

### Code

**Volumetric data generation**

Matlab functions and scripts to generate volumetric labels from manual traces using a topology preserving fast marching algorithm can be found [here](https://github.com/ogliko/patchseq-autorecon).

**Axonal segmentation**

Automated segmentation of axons. This project is built on previous projects. 
Original github repositories: 
- [Neurotorch](https://github.com/jgornet/NeuroTorch)
- [DeepEM](https://github.com/seung-lab/DeepEM)

**Segmentation model**

Code for training a neuron network model to perform axonal segmentation is in the `src/ac_segmentation/training` section of this repository. 

***To train:***
```
python train.py \
--ckpt_dir DIRECTORY_TO_SAVE_CHECKPOINTS \
--log_dir DIRECTORY_TO_SAVE_LOGS \
--json_dir DIRECTORY_WITH_INPUT_AND_LABEL_JSON_FILES \
--eps EPSILON_PARAMETER_FOR_ADAM_OPTIMIZER \
--epochs NUMBER_OF_EPOCHS
```
***Note***: The filepaths in the inputs' and labels' json files are examples and need to be changed.

Trained model is in `demos/model_files/segmentation` section.
