import numpy as np
import tifffile as tif
import os
import glob
import argparse

import ac_segmentation.neurotorch.datasets.datatypes
import ac_segmentation.neurotorch.datasets.dataset
import ac_segmentation.neurotorch.nets.RSUNet
import ac_segmentation.neurotorch.core.predictor
import ac_segmentation.neurotorch.datasets.filetypes

Predictor = ac_segmentation.neurotorch.core.predictor.Predictor
Vector = ac_segmentation.neurotorch.datasets.datatypes.Vector
BoundingBox = ac_segmentation.neurotorch.datasets.datatypes.BoundingBox
Array = ac_segmentation.neurotorch.datasets.dataset.Array
RSUNet = ac_segmentation.neurotorch.nets.RSUNet.RSUNet
TiffVolume = ac_segmentation.neurotorch.datasets.filetypes.TiffVolume

def validate(checkpoint, test_dir, out_dir, bb, num_parts):
    # Initialize the U-Net architecture
    net = RSUNet()

    offset = 0
    for n in range(num_parts):
        bbn = BoundingBox(Vector(bb[0], bb[1], bb[2]), Vector(bb[3], bb[4], bb[5+n]))
        print(n, bbn)
        print('offset', offset) 
        if num_parts==1:
            filename = 'inputs' + '.tif'
        else:
            filename = 'inputs' + str(n) + '.tif'
        print(os.path.join(test_dir, filename)) 
        
        with TiffVolume(os.path.join(test_dir, filename), bbn) as inputs:              
            # Predict
            predictor = Predictor(net, checkpoint, gpu_device=0)
            output_volume = Array(-np.inf*np.ones(inputs.getBoundingBox().getNumpyDim(), dtype=np.float32))
            print('bb0', inputs.getBoundingBox())
            predictor.run(inputs, output_volume)
                            
            # Convert to probability map and save
            probability_map = 1/(1+np.exp(-output_volume.getArray()))
            print('probability_map', type(probability_map), probability_map.shape, probability_map.dtype)
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            for i in range(probability_map.shape[0]): # save 8-bit version as multiple tif files
                tif.imsave(os.path.join(out_dir,'%04d.tif'%(i+offset)), np.uint8(255*probability_map[i,:,:]))  
        offset = offset + bb[5+n]          

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', '-c', type=str, help='path to checkpoint')
    parser.add_argument('--test_dir', '-v', type=str, help='directory of validation/test data')
    parser.add_argument('--out_dir', '-o', type=str, help='results directory path')
    parser.add_argument('--bb', '-b', nargs='+', type=int, help='bounding box')
    parser.add_argument('--num_parts', '-n', type=int, help='number of parts to divide volume')
    args = parser.parse_args()
    validate(args.ckpt, args.test_dir, args.out_dir, args.bb, args.num_parts)
