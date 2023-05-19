### ac_segmentation

This repository contains code for reconstructing axons from light microscopy images.

### Code

**Axonal reconstruction pipeline**

Automated pipeline combines segmentation of raw image stacks with post-processing to produce a swc file including radius calculation and topological correction. 
Original github repo [here](https://github.com/ogliko/patchseq-autorecon). 

**Volumetric data generation**

Matlab functions and scripts to generate volumetric labels from manual traces using a topology preserving fast marching algorithm.
Original github repo [here](https://github.com/rhngla/topo-preserve-fastmarching).

**Axonal segmentation**

Automated segmentation of axons. This project is built on previous projects. 
Original github repositories: 
- [Neurotorch](https://github.com/jgornet/NeuroTorch)
- [DeepEM](https://github.com/seung-lab/DeepEM)

### Installation

first create an environment and install numpy and scikit-build (some external dependencies require these for building)
`conda create -n ac_seg numpy scikit-build`

Then in the top level of this directory install with:
`pip install .`