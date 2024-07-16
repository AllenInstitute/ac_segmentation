import pytest
import numpy as np
from ac_segmentation.utils.tensorstore import create_tensor
from ac_segmentation.utils.io import gzip_array


@pytest.fixture(scope="session")
def zarr_file(tmp_path_factory):
    fn = str(tmp_path_factory.mktemp("temp"))
    arr = np.random.choice([0, 1], size=(1,1,100,100,100), p=[.7, .3])
    tensor = create_tensor(fpath=fn+"test_zarr.zarr", arr_shape=[1,1,100,100,100], arr=arr, chunk_shape = [1,1,64,64,64], dtype = 'uint16', fill_value=0)
    return fn
    
    
@pytest.fixture(scope="session")
def npy_file(tmp_path_factory):
    fn = str(tmp_path_factory.mktemp("temp"))
    arr = np.random.choice([0, 1], size=(100,100,100), p=[.7, .3])
    gzip_array(fn+"test_zarr.npy.gz", arr)
    return fn