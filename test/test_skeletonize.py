import os
from ac_segmentation.utils.tensorstore import open_tensor, AWS_Parameters, create_kvstore, create_tensor
from ac_segmentation.methods.skeletonize_array import TS_skeletonize_volume


def test_skel(zarr_file):
    #run(probability_input_path=npy_file+"test_zarr.npy.gz", 
        #skeleton_output_path=npy_file+"output_skel.swcs.tar.gz",probability_threshold=0.05,label_size_threshold=80)
    input_arr = open_tensor(zarr_file+"test_zarr.zarr", bytes_limit= 100_000_000, driver='zarr')    
    TS_skeletonize_volume(input_arr, chunk_size=[100,100,100], n_jobs=4, overlap=4)
            
            
            