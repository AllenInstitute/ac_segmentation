import gunpowder as gp
from gunpowder.ext import ZarrFile
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from gunpowder.provider_spec import ProviderSpec
from zarr._storage.store import BaseStore
from zarr import N5Store, N5FSStore

import numpy as np
from collections.abc import MutableMapping
from typing import Union
import warnings
import logging
import copy
import torch 
import os
from datetime import datetime
import itertools
from functools import lru_cache

from ac_segmentation.utils.tensorstore import open_tensor, AWS_Parameters, create_kvstore, create_tensor
from ac_segmentation.utils.preprocess import lut_preprocess_array_minmax


logger = logging.getLogger(__name__)



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
    
    

class TensorStoreSource(gp.ZarrSource):

    def __init__(self, tensorstore=None, array_specs=None, channels_first=True):
        if array_specs is None:
            self.array_specs = {}
        else:
            self.array_specs = array_specs

        self.channels_first = channels_first
        self.tensorstore = tensorstore

    def _get_offset(self, dataset):
        if "offset" not in dataset.attrs:
            return None

        if self._rev_metadata():
            return Coordinate(dataset.attrs["offset"][::-1])
        else:
            return Coordinate(dataset.attrs["offset"])

    def _rev_metadata(self):
        with ZarrFile(self.store, mode="a") as store:
            return isinstance(store.chunk_store, N5Store) or isinstance(store.chunk_store, N5FSStore)

    def setup(self):
        for array_key, tensorstore in self.tensorstore.items():
            spec = self.__read_spec(array_key, tensorstore)
            self.provides(array_key, spec, tensorstore)

    def provides(self, key, spec, tensorstore):
        """Introduce a new output provided by this :class:`BatchProvider`."""
        name = 'TensorStoreSource[' + str(tensorstore.kvstore.path) + ']'
        logger.debug("Current spec of %s:\\n%s", name, self.spec)

        if self.spec is None:
            self._spec = ProviderSpec()

        assert (key not in self.spec), "Node %s is trying to add spec for %s, but is already provided." % (type(self).__name__, key)

        self.spec[key] = copy.deepcopy(spec)
        self.provided_items.append(key)

        logger.debug("%s provides %s with spec %s", name, key, spec)

    def provide(self, request):
        timing = Timing(self)
        timing.start()

        batch = Batch()

        for akey, tensorstore in self.tensorstore.items():
            for array_key, request_spec in request.array_specs.items():
                logger.debug("Reading %s in %s...", array_key, request_spec.roi)

                voxel_size = self.spec[array_key].voxel_size

                # scale request roi to voxel units
                dataset_roi = request_spec.roi / voxel_size

                # shift request roi into dataset
                dataset_roi = dataset_roi - self.spec[array_key].roi.offset / voxel_size

                # create array spec
                array_spec = self.spec[array_key].copy()
                array_spec.roi = request_spec.roi
                
                # add array to batch
                batch.arrays[array_key] = Array(self.__read(tensorstore, dataset_roi), array_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read(self, data_file, roi):
        c = len(data_file.shape) - self.ndims

        if self.channels_first:
            array = data_file[(slice(None),) * c + roi.to_slices()].read().result()
        else:
            array = data_file[roi.to_slices() + (slice(None),) * c].read().result()
            array = np.transpose(array, axes=[i + self.ndims for i in range(c)] + list(range(self.ndims)))

        return array

    def __read_spec(self, array_key, tensorstore):
        dataset = tensorstore

        if array_key in self.array_specs:
            spec = self.array_specs[array_key].copy()
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:
            voxel_size = Coordinate((1,) * len(dataset.shape))
            logger.warning(
                "WARNING: File %s does not contain resolution information for %s, voxel size has been set to %s. This might not be what you want.",
                tensorstore.kvstore.path,
                array_key,
                spec.voxel_size,
            )

        spec.voxel_size = voxel_size
        self.ndims = len(spec.voxel_size)

        if spec.roi is None:
            #offset = self._get_offset(dataset) RETURN TO THIS!
            offset = None
            if offset is None:
                offset = Coordinate((0,) * self.ndims)

            if self.channels_first:
                shape = Coordinate(dataset.shape[-self.ndims :])
            else:
                shape = Coordinate(dataset.shape[: self.ndims])

            spec.roi = Roi(offset, shape * spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == dataset.dtype.name, (
                "dtype %s provided in array_specs for %s, but differs from dataset dtype %s"
                % (self.array_specs[array_key].dtype, array_key, dataset.dtype.name)
            )
        else:
            spec.dtype = dataset.dtype.name

        if spec.interpolatable is None:
            spec.interpolatable = np.issubdtype(spec.dtype, np.floating) or (spec.dtype == np.uint8)
            logger.warning(
                "WARNING: You didn't set 'interpolatable' for %s. Based on the dtype %s, it has been set to %s. This might not be what you want.",
                array_key,
                spec.dtype,
                spec.interpolatable,
            )

        return spec

    def name(self):
        return 'TensorStoreSource[' + list(self.tensorstore.values())[0].kvstore.path + ']'
        
        
class ContrastAdjust(gp.BatchFilter):
    def __init__(self, input_key, output_key, int_range=None, version='range'):
        self.input_key = input_key
        self.output_key = output_key
        self.int_range = int_range
        self.version = version

    def setup(self):
        pass

    def prepare(self, request):
        # Ensure the input array is requested
        deps = gp.BatchRequest()
        deps[self.input_key] = request[self.output_key].copy()
        return deps

    def process(self, batch, request):
        # Get the input data
        input_data = batch[self.input_key].data

        # Apply the contrast adjustment function with the specified parameters
        if self.int_range:
            r1,r2 = self.int_range
            if self.version == 'range':
                adjusted_data = lut_preprocess_array_minmax(input_data, r1, r2)

            if self.version == 'percentile':
                p1, p2 = np.percentile(input_data, self.int_range)
                scale = 1.0 / (p2 - p1) if p2 > p1 else 1.0
                adjusted_data = np.clip((input_data - p1) * scale, 0, 1)
                adjusted_data = (adjusted_data * 255).astype(str(input_data.dtype))

        # Create a new batch with the adjusted data
        spec = batch[self.input_key].spec.copy()
        spec.roi = request[self.output_key].roi.copy()

        # Create a new array
        adjusted_array = gp.Array(adjusted_data, spec)

        # Store it in the batch
        batch = gp.Batch()
        batch[self.output_key] = adjusted_array

        return batch

class ApplyModel(gp.BatchFilter):
    def __init__(self, model, input_key, ts_array, device):
        self.model = model
        self.input_key = input_key
        self.ts_array = ts_array
        self.write_objects = []
        self.device = device

    def process(self, batch, request):
        # Get the input data
        input_data = batch[self.input_key].data
        og_shape = input_data.shape

        # CHECK THISSSSS!! 
        if input_data.dtype != np.int16:
            input_data = input_data.astype(np.int16)
        if len(input_data.shape)<5:
            x,y,z = input_data.shape
            input_data = input_data.reshape(1, 1, x, y, z)
            
        # Convert input data to a tensor
        input_tensor = torch.from_numpy(input_data).float().to(self.device)

        # Run the model
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # Convert output tensor to probability map
        output_data = output_tensor[0].data.cpu()
        output_data = torch.special.expit(output_data).numpy()[0,0,:,:,:]

        # Write predictions to TensorStore
        roi = batch.arrays[self.input_key].spec.roi
        offset = roi.get_begin()[-3:]

        arr_blend = self.ts_array[offset[0]:offset[0] + output_data.shape[0],
                        offset[1]:offset[1] + output_data.shape[1],
                        offset[2]:offset[2] + output_data.shape[2]].read().result()
        
        if np.isneginf(arr_blend).any():
            arr_blend[arr_blend == -np.inf] = 0
        arr_blend += output_data
        
        write_obj = self.ts_array[offset[0]:offset[0] + output_data.shape[0],
                        offset[1]:offset[1] + output_data.shape[1],
                        offset[2]:offset[2] + output_data.shape[2]].write(arr_blend)
        
        self.write_objects.append(write_obj) 

    def get_write_objects(self):
        return self.write_objects

    def clear_write_objects(self):
        self.write_objects = []
        
        
class ContrastAdjustWrite(gp.BatchFilter):
    def __init__(self, input_key, output_key, input_arr, output_arr, int_range=None, version='range'):
        self.input_key = input_key
        self.output_key = output_key
        self.int_range = int_range
        self.version = version
        self.out_array = output_arr
        self.in_array = input_arr
        self.write_objects = []

    def setup(self):
        pass

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.input_key] = request[self.output_key].copy()
        return deps

    def process(self, batch, request):
        roi = batch.arrays[self.input_key].spec.roi
        start, end = list(roi.begin[-3:]), list(roi.end[-3:])
        x1, y1, z1 = start
        x2, y2, z2 = end

        input_data = batch[self.input_key].data

        p1, p2 = np.percentile(input_data, self.int_range)
        
        if np.any(input_data) == True:
            if len(self.in_array.shape) ==5:
                input_data = input_data[0,0,:,:,:]
            scale = 1.0 / (p2 - p1) if p2 > p1 else 1.0
            output_data = np.clip((input_data - p1) * scale, 0, 1)
            output_data = (output_data * 255)

            if len(self.in_array.shape) ==5:
                arr2 = self.out_array[0,0,x1:x2, y1:y2, z1:z2].read().result()
                if (output_data.shape[-3:] == (output_data.shape[2],) * len(output_data.shape[-3:])) == True: ###can i remove this???
                    output_data = perimeter_weighted_blend(arr2, output_data, depth=.9).astype('uint8')
                    write_obj = self.out_array[:,:,x1:x2, y1:y2, z1:z2].write(output_data[None, None, :])
                self.write_objects.append(write_obj)
            else:
                arr2 = self.out_array[x1:x2, y1:y2, z1:z2].read().result()
                if (output_data.shape == (output_data.shape[0],) * len(output_data.shape)) == True:
                    output_data = perimeter_weighted_blend(arr2, output_data, depth=.9).astype('uint8')
                    write_obj = self.out_array[x1:x2, y1:y2, z1:z2].write(output_data)
                self.write_objects.append(write_obj)

    def get_write_objects(self):
        return self.write_objects

    def clear_write_objects(self):
        self.write_objects = []
        

