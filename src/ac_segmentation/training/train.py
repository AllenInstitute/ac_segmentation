from ac_segmentation.neurotorch.datasets.specification import JsonSpec
from ac_segmentation.neurotorch.datasets.dataset import PooledVolume
from ac_segmentation.neurotorch.datasets.datatypes import (BoundingBox, Vector)
from ac_segmentation.neurotorch.datasets.filetypes import TiffVolume
from ac_segmentation.neurotorch.core.trainer import Trainer
from ac_segmentation.neurotorch.nets.RSUNet import RSUNet
import os
import numpy as np
import torch
import torch.optim as optim
import argparse
import random

# default values for argparse
CKPT = 'None'
CKPT_DIR = 'run1'
LOG_DIR = 'logs'
JSON_DIR = 'data'
EPS_DEFAULT = 1e-1
EPOCHS = 1

def train(ckpt, ckpt_dir, log_dir, json_dir, eps, epochs, augmentation):
    inputs_list = [f for f in os.listdir(json_dir) if 'inputs' in f]
    inputs_list.sort()
    labels_list = [f for f in os.listdir(json_dir) if 'labels' in f]
    labels_list.sort()    

    # Initialize network and json specification.
    net = RSUNet()
    json_spec = JsonSpec()

    # Define experiment name from arguments
    exp_name = str(json_dir.split('data/')[1]) + '_' + str(eps) + '_' + str(ckpt_dir)

    # Define checkpoints directory
    ckpt_dir = os.path.join('checkpoints', exp_name)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    # Define log directory
    log_dir = os.path.join(log_dir, exp_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        
    spec1 = [json_spec.parse(os.path.join(json_dir, inputs_list[j]))[0] 
             for j in range(len(inputs_list))]
    spec2 = [json_spec.parse(os.path.join(json_dir, labels_list[j]))[0] 
             for j in range(len(inputs_list))]   
    
    validation_split = 0.001
    inputs_vol = []
    labels_vol = []
    inputs_vol_val = []
    labels_vol_val = []  

    for s in range(len(spec1)):
        volume1 = []
        volume2 = []
        spec = [spec1[s]]
        inputs = create_volume(spec) 
        spec = [spec2[s]]
        labels = create_volume(spec) 

        for n in range(len(inputs)):
            if not (labels[n].getArray() == 0).all():
                volume1.append(inputs[n].getArray().astype(np.uint8))
                volume2.append(labels[n].getArray().astype(np.uint8))

        valid_indexes = np.arange(len(volume1))
        np.random.seed(0)
        random_idx = np.random.permutation(valid_indexes)
        val_idx = random_idx[int(len(valid_indexes)*(1-validation_split)):].copy()
        volume1_val = [volume1[ind] for ind in val_idx] 
        volume2_val = [volume2[ind] for ind in val_idx] 

        for ind in sorted(val_idx, reverse=True): 
            del volume1[ind]
            del volume2[ind]

        inputs_vol = inputs_vol + volume1
        labels_vol = labels_vol + volume2
        inputs_vol_val = inputs_vol_val + volume1_val
        labels_vol_val = labels_vol_val + volume2_val
       
    inputs_vol = [inputs_vol, inputs_vol_val]
    labels_vol = [labels_vol, labels_vol_val]

    # Initialize optimizer with updated epsilon parameter
    optimizer = optim.Adam(net.parameters(), eps=eps, amsgrad=True)

    # Initialize trainer
    if ckpt == 'None':
        trainer = Trainer(net, inputs_vol, labels_vol, augmentation, checkpoint_dir=ckpt_dir, checkpoint_period=5000, 
                          logger_dir=log_dir, max_epochs=epochs, gpu_device=0, optimizer=optimizer)       
    else:
        trainer = Trainer(net, inputs_vol, labels_vol, augmentation, checkpoint_dir=ckpt_dir, checkpoint_period=5000, 
                          logger_dir=log_dir, checkpoint=ckpt, max_epochs=epochs, gpu_device=0, optimizer=optimizer) 
                          

    # begin training
    trainer.run_training()  

def create_volume(volume_spec, stack_size=33, iteration_size=BoundingBox(Vector(0, 0, 0), Vector(128, 128, 128)),
                  stride=Vector(16, 16, 16)):
    pooled_volume = PooledVolume(stack_size=stack_size, iteration_size=iteration_size, stride=stride)
    for item in volume_spec:
        filename = os.path.abspath(item["filename"])
        edges = item["bounding_box"]
        bounding_box = BoundingBox(Vector(*edges[0]), Vector(*edges[1]))
        volume = TiffVolume(filename, bounding_box, iteration_size=iteration_size, stride=stride)
        pooled_volume.add(volume)
    return pooled_volume        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', '-ck', type=str, default=CKPT, help='path to checkpoint')
    parser.add_argument('--ckpt_dir', '-c', type=str, default=CKPT_DIR)
    parser.add_argument('--log_dir', '-l', type=str, default=LOG_DIR)
    parser.add_argument('--json_dir', '-j', type=str, default=JSON_DIR)
    parser.add_argument('--eps', '-e', type=float, default=EPS_DEFAULT)
    parser.add_argument('--epochs', '-ep', type=int, default=EPOCHS)
    parser.add_argument('--augmentation', '-a', type=int, default=0, help='1-true 0-false')
    args = parser.parse_args()
    train(args.ckpt, args.ckpt_dir, args.log_dir, args.json_dir, args.eps, args.epochs, args.augmentation)
