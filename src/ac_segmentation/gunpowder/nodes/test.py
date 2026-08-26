import configparser
from typing import Optional
import pandas as pd
import numpy as np
import tensorstore as ts
import os
import json
import boto3
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing_extensions import Self


def split_s3_path(s3_path):
    if 'https' in s3_path:
        path_parts=s3_path.replace("https://","").split("/")
        bucket=path_parts.pop(0).split(".s3")[0]
        key="/".join(path_parts)
    else:
        path_parts=s3_path.replace("s3://","").split("/")
        bucket=path_parts.pop(0)
        key="/".join(path_parts)
    return bucket, key

class AWS_Parameters:
    entries: dict[int, tuple[str, str]]
    temp_dir: TemporaryDirectory[str]
    credentials_file_path: Path
    @classmethod
    @lru_cache
    def singleton(cls) -> "Self":
        return cls()
        
    def __init__(self, profile=None, region=None, endpoint_url=None):
        self.entries = {}
        self.temp_dir = TemporaryDirectory()
        self.credentials_file_path = Path(self.temp_dir.name) / "aws_credentials"
        self.credentials_file_path.touch()
        #create session
        session = boto3.Session(profile_name=profile, region_name=region)
        if endpoint_url:
            self.endpoint_url=endpoint_url
        self.profile=session.profile_name
        self.region=session.region_name
    def _dump_credentials(self) -> None:
        self.credentials_file_path.write_text(
            "\n".join(
                [
                    f"[{self.profile}]\naws_access_key_id = {access_key_id}\naws_secret_access_key = {secret_access_key}\n"
                    for key_hash, (
                        access_key_id,
                        secret_access_key,
                    ) in self.entries.items()
                ]
            )
        )
    def add_credentials(self, access_key_id: str, secret_access_key: str) -> dict[str, str]:
        key_tuple = (access_key_id, secret_access_key)
        key_hash = hash(key_tuple)
        self.entries[key_hash] = key_tuple
        self._dump_credentials()
        self.credential_file = {
            "profile": f"profile-{key_hash}",
            "filename": str(self.credentials_file_path),
            "metadata_endpoint": "",
        }


def create_kvstore(fpath, store, AWS_param=None):
    """Creates the kvstore configuration based on the input parameters.

    Args:
        fpath (str): Path to the tensorstore file or S3 URL.
        store (str): Type of store ('file' or 's3').
        AWS_param (Optional[dict]): AWS credentials and parameters (only used for S3).

    Returns:
        dict: The kvstore configuration.
    """
    kvstore = {"driver": store, "path": fpath}
    
    if store == 's3':
        # Parse the S3 URL into bucket and path
        bucket, path = split_s3_path(fpath)
        kvstore = {"driver": "s3", "bucket": bucket, "path": path}
        
        if AWS_param:
            kvstore.update({"aws_region": AWS_param.region})
            if hasattr(AWS_param, "endpoint_url"):
                kvstore.update({"endpoint": AWS_param.endpoint_url})
            
            # Handle credentials
            cred = {"aws_credentials": {"profile": AWS_param.profile}}
            if hasattr(AWS_param, "credential_file"):
                cred = {"aws_credentials": {
                    "profile": AWS_param.profile,
                    "filename": AWS_param.credential_file['filename']
                }}
            kvstore.update(cred)
    
    return kvstore
    
    
def open_tensor(fpath=None, kvstore=None, driver='zarr', bytes_limit=100_000_000):
    """Open a tensorstore object.

    Args:
        fpath (str): Path to the tensorstore file or S3 URL.
        driver (str): Type of file (e.g., 'zarr', 'n5', 'precomputed').
        kvstore (dict, optional): Pre-constructed kvstore configuration.
        bytes_limit (int): Memory limit for in-memory cache in bytes (default 100MB).

    Returns:
        tensorstore.Dataset: The opened tensorstore dataset.
    """
    # If kvstore is not provided, create it from fpath
    if kvstore is None:
        kvstore = create_kvstore(fpath, store='file', AWS_param=None)

    # Check if zarr v3
    if 'zarr' in driver:
        # Load the tensorstore array with cache configuration
        try:
            dataset_future = ts.open({
                'driver': 'zarr',
                'kvstore': kvstore,
                'context': {
                    'cache_pool': {
                        'total_bytes_limit': bytes_limit
                    }
                },
                'recheck_cached_data': 'open',
            })
            return dataset_future.result()
    
        except:
            dataset_future = ts.open({
                'driver': 'zarr3',
                'kvstore': kvstore,
                'context': {
                    'cache_pool': {
                        'total_bytes_limit': bytes_limit
                    }
                },
                'recheck_cached_data': 'open',
            })
            return dataset_future.result()
            
    else:
         dataset_future = ts.open({
                'driver': driver,
                'kvstore': kvstore,
                'context': {
                    'cache_pool': {
                        'total_bytes_limit': bytes_limit
                    }
                },
                'recheck_cached_data': 'open',
            })
         return dataset_future.result()
        



import logging
import os
import zarr
import numcodecs
numcodecs.blosc.use_threads = False

from gunpowder.nodes import BatchFilter  # noqa
from gunpowder.batch_request import BatchRequest  # noqa
from gunpowder.roi import Roi  # noqa
from gunpowder.coordinate import Coordinate  # noqa
from gunpowder.ext import ZarrFile  # noqa

logger = logging.getLogger(__name__)


import logging
import os
import zarr
import numcodecs
numcodecs.blosc.use_threads = False

from gunpowder.nodes import BatchFilter  # noqa
from gunpowder.batch_request import BatchRequest  # noqa
from gunpowder.roi import Roi  # noqa
from gunpowder.coordinate import Coordinate  # noqa
from gunpowder.ext import ZarrFile  # noqa

logger = logging.getLogger(__name__)


class TensorStoreWrite(BatchFilter):
    '''Assemble arrays of passing batches in one zarr container. This is useful
    to store chunks produced by :class:`Scan` on disk without keeping the
    larger array in memory. The ROIs of the passing arrays will be used to
    determine the position where to store the data in the dataset.

    Args:

        dataset_names (``dict``, :class:`ArrayKey` -> ``string``):

            A dictionary from array keys to names of the datasets to store them
            in.

        output_dir (``string``):

            The directory to save the zarr container. Will be created, if it does
            not exist.

        output_filename (``string``):

            The output filename of the container. Will be created, if it does
            not exist, otherwise data is overwritten in the existing container.

        compression_type (``string`` or ``int``):

            Compression strategy.  Legal values are ``gzip``, ``szip``,
            ``lzf``. If an integer between 1 and 10, this indicates ``gzip``
            compression level.

        dataset_dtypes (``dict``, :class:`ArrayKey` -> data type):

            A dictionary from array keys to datatype (eg. ``np.int8``). If
            given, arrays are stored using this type. The original arrays
            within the pipeline remain unchanged.

        chunks (``tuple`` of ``int``, or ``bool``):
            Chunk shape for output datasets. Set to ``True`` for auto-chunking,
            set to ``False`` to obtain a chunk equal to the dataset size.
            Defaults to ``True``.
    '''

    def __init__(
            self,
            dataset_names,
            output_dir='.',
            output_filename='output.hdf',
            dataset_dtypes=None,
            chunks=True):

        self.dataset_names = dataset_names
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.compression_type = 'blosc'
        self.full_path = os.path.join(output_filename, str(list(dataset_names.values())[0]))
        
        if dataset_dtypes is None:
            self.dataset_dtypes = {}
        else:
            self.dataset_dtypes = dataset_dtypes
        self.chunks = chunks

        self.dataset_offsets = {}

    def setup(self):
        for key in self.dataset_names.keys():
            self.updates(key, self.spec[key])
        self.enable_autoskip()

    def prepare(self, request):
        deps = BatchRequest()
        for key in self.dataset_names.keys():
            deps[key] = request[key]
        return deps

    def init_datasets(self, batch):

        filename = os.path.join(self.output_dir, self.output_filename)
        logger.debug("Initializing container %s", filename)

        try:
            os.makedirs(self.output_dir)
        except BaseException:
            pass

        for (array_key, dataset_name) in self.dataset_names.items():

            logger.debug("Initializing dataset for %s", array_key)

            assert array_key in self.spec, (
                "Asked to store %s, but is not provided upstream." % array_key)
            assert array_key in batch.arrays, (
                "Asked to store %s, but is not part of batch." % array_key)

            array = batch.arrays[array_key]
            dims = array.spec.roi.dims
            batch_shape = array.data.shape

            data_file = self._open_file(filename)
            offset = Coordinate((0,0,0,0,0))



            logger.debug(
                    "%s (%s in %s) has offset %s",
                    array_key,
                    dataset_name,
                    filename,
                    offset)
            self.dataset_offsets[array_key] = offset

    def process(self, batch, request):

        filename = os.path.join(self.output_dir, self.output_filename)

        if not self.dataset_offsets:
            self.init_datasets(batch)

        dataset =  self._open_file(filename) 

        for (array_key, dataset_name) in self.dataset_names.items():
            try:
                array_roi = batch.arrays[array_key].spec.roi
                voxel_size = self.spec[array_key].voxel_size
                dims = array_roi.dims
                channel_slices = (slice(None),) * \
                    max(0, len(dataset.shape) - dims)

                dataset_roi = Roi(
                    self.dataset_offsets[array_key],
                    Coordinate(dataset.shape[-dims:]) * voxel_size)
                common_roi = array_roi.intersect(dataset_roi)

                dataset_voxel_roi = (
                    common_roi - self.dataset_offsets[array_key]) // voxel_size
                dataset_voxel_slices = dataset_voxel_roi.to_slices()
                array_voxel_roi = (
                    common_roi - array_roi.get_offset()) // voxel_size
                array_voxel_slices = array_voxel_roi.to_slices()

                logger.debug(
                    "writing %s to voxel coordinates %s" % (
                        array_key,
                        dataset_voxel_roi))

                data = batch.arrays[array_key].data[channel_slices +
                                                        array_voxel_slices]

                dataset[channel_slices + dataset_voxel_slices]=data.astype(dataset.dtype.name)

            except ValueError as e:
                logger.warning(f"Array '{array_key}' failed to write: {e}")
                

    def _get_voxel_size(self, dataset):

        if 'resolution' not in dataset.attrs:
            return None

        if self.output_filename.endswith('.n5'):
            return Coordinate(dataset.attrs['resolution'][::-1])
        else:
            return Coordinate(dataset.attrs['resolution'])

    def _get_offset(self, dataset):

        if 'offset' not in dataset.attrs:
            return None

        if self.output_filename.endswith('.n5'):
            return Coordinate(dataset.attrs['offset'][::-1])
        else:
            return Coordinate(dataset.attrs['offset'])

    def _set_voxel_size(self, dataset, voxel_size):

        if self.output_filename.endswith('.n5'):
            dataset.attrs['resolution'] = voxel_size[::-1]
        else:
            dataset.attrs['resolution'] = voxel_size

    def _set_offset(self, dataset, offset):

        if self.output_filename.endswith('.n5'):
            dataset.attrs['offset'] = offset[::-1]
        else:
            dataset.attrs['offset'] = offset

    def _open_file(self, filename):
        return open_tensor(self.full_path)
        
    
            
            
            
import numpy as np
import gunpowder as gp

import numpy as np
from collections.abc import MutableMapping
from typing import Union
import warnings
import logging
import copy
import torch 
import os
import itertools
from functools import lru_cache

def no_neg(value):
    return value if value >= 0 else 0

@lru_cache(maxsize=40)
def make_mask(shape, depth=1):
    min_dim = min(shape)
    out = np.ones((min_dim,min_dim,min_dim))
    layers = int(min_dim*(depth/2))
    intervals = np.linspace(0, .8, layers)
    
    for ind, inter in enumerate(intervals):
        out[ind,:,:] = inter
        out[min_dim-1-ind,:,:] = inter
    
    y_swap = np.transpose(out.copy(), (1, 0, 2))  
    z_swap = np.transpose(out.copy(), (2, 1, 0))

    out = np.minimum(out, y_swap.copy())
    out = np.minimum(out,  z_swap.copy())

    if (shape == (shape[0],) * len(shape)) == False:
        x1, y1, z1 = out.shape
        x2, y2, z2 = shape
        
        x = np.linspace(0, x1 - 2, x2).astype(int)
        y = np.linspace(0, y1 - 2, y2).astype(int)
        z = np.linspace(0, z1 - 2, z2).astype(int)
        out = ((out[np.ix_(x, y, z)]))

    return out

def perimeter_weighted_blend(array1, array2, depth=.5):
    weight_map = make_mask(array1.shape, depth)
    return (array1 * (1 - weight_map) + array2 * (weight_map))


class DummyContrastNode(gp.BatchFilter):
    def __init__(self, array_a, perc_range=[10,99]):
        super().__init__()
        self.array_a = array_a
        self.perc_range = perc_range
        self.write_objects = []

        
    def process(self, batch, request):
        raw_data = batch[self.array_a].data
        in_shape = raw_data.shape

        p1, p2 = np.percentile(raw_data, self.perc_range)

        if np.any(raw_data):
            if len(in_shape) == 5:
                input_data = raw_data[0, 0, :, :, :]
            else:
                input_data = raw_data

            scale = 1.0 / (p2 - p1) if p2 > p1 else 1.0
            output_data = np.clip((input_data - p1) * scale, 0, 1)
            output_data = (output_data * 255)
                

            if len(in_shape) == 5:
                output_data = output_data[np.newaxis, np.newaxis, ...]
                                

            self.write_objects.append(output_data)

        else:
            pass
            
            
