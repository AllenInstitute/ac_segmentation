import os
from ac_segmentation.postprocess.skeletonize_array_uint8_prob import run


def test_seg_skel_uint8(npy_file):
    run(probability_input_path=npy_file+"test_zarr.npy.gz", 
        skeleton_output_path=npy_file+"output_skel.swcs.tar.gz",probability_threshold=0.05,label_size_threshold=80)
            
            
            