### ac_segmentation

This repository contains code for segmenting and skeletonizing data from light microscopy images. Demo notebooks can be found in src/demos/.


### Nested Packages

**Volumetric data generation**

Matlab functions and scripts to generate volumetric labels from manual traces using a topology preserving fast marching algorithm can be found [here](https://github.com/ogliko/patchseq-autorecon).

**Axonal segmentation**

Automated segmentation of axons. This project is built on previous projects. 
Original github repositories: 
- [Neurotorch](https://github.com/jgornet/NeuroTorch)
- [DeepEM](https://github.com/seung-lab/DeepEM)
- [Gunpowder](https://github.com/funkelab/gunpowder.git)


### Methods

**Segmentation**

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

Trained model is in `src/demos/model_files/segmentation` section.


***To test:***
```
python src/ac_segmentation/methods/segment_array.py \
--input_path DIRECTORY_TO_INPUT_ZARR_VOLUME  \
--output_path DIRECTORY_TO_OUTPUT_ZARR_VOLUME \
--weights_file DIRECTORY_TO_WEIGHTS_FILE \
--mask_path DIRECTORY_TO_MASK_ZARR_VOLUME \
--dsfactor MASK_DOWNSAMPLE_FACTOR \
--filter_max_intensity MAXIMUM_THREHSOLD_INTENSITY
```

**Skeletonization**
```
python src/ac_segmentation/methods/skeletonize_array.py \
--input_path DIRECTORY_TO_INPUT_ZARR_VOLUME \
--skeleton_output DIRECTORY_TO_OUTPUT_SWC_VOLUME \
--probability_threshold MINIMUM_PROBABILITY_VALUE \
--label_size_threshold MINIMUM_CONNECTED_COMPONENT_VALUE
```

**Contrast Equalization**
```
python src/ac_segmentation/methods/equalize_array.py \
--input_path DIRECTORY_TO_INPUT_ZARR_VOLUME  \
--output_path DIRECTORY_TO_OUTPUT_ZARR_VOLUME \
--mask_path DIRECTORY_TO_MASK_ZARR_VOLUME \
--dsfactor MASK_DOWNSAMPLE_FACTOR \
```

**Multi-Tile Image Fusion**
```
python src/ac_segmentation/methods/fuse_volume.py \
--in_paths PATH_TO_TILES_LIST_FILE \
--translations PATH_TO_TRANSLATIONS_LIST_FILE \
--output_path PATH_TO_OUTPUT_ZARR_VOLUME \
--mip MULTISCALE_LEVEL \
--blend AVERAGE_BLENDING_FLAG
```
