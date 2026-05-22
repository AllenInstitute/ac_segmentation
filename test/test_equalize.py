import pytest
import os
import glob
from ac_segmentation.utils.tensorstore import open_tensor, AWS_Parameters, create_kvstore, create_tensor
from ac_segmentation.methods.equalize_array import adjust_contrast_gunpowder


def test_equalize(zarr_file):
    path = os.path.normpath(os.path.dirname(os.path.realpath(__file__)) + os.sep + os.pardir)
    weights = glob.glob(f"{path}/**/{'best.ckpt'}", recursive=True)[0]
    
    input_arr = open_tensor(zarr_file+"test_zarr.zarr", bytes_limit= 100_000_000, driver='zarr')
    output_arr = create_tensor(fpath=os.path.join(zarr_file, "output_equalize"), arr_shape=input_arr.shape, dtype='uint8', chunk_shape=[1, 1, 64, 64, 64], driver='zarr3', codecs=None)

    adjust_contrast_gunpowder(input_arr, output_arr, iter_size=(64,64,64), batch_size=5, preprocess={'method':'percentile','values':[96,97]}, add_margin=0, depth=.7)