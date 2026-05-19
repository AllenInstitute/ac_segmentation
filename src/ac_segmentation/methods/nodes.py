# Add cutout capability
import matplotlib.pyplot as plt


#from zarr._storage.store import BaseStore
#from zarr import N5Store, N5FSStore

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

import tensorstore as ts

from ac_segmentation.utils.tensorstore import open_tensor, AWS_Parameters, create_kvstore, create_tensor
from ac_segmentation.utils.preprocess import lut_preprocess_array_minmax
from ac_segmentation.methods.bump_mask import *


from ac_segmentation.gunpowder.array_spec import ArraySpec
from ac_segmentation.gunpowder.array import ArrayKey
from ac_segmentation.gunpowder.coordinate import Coordinate
from ac_segmentation.gunpowder.batch_request import BatchRequest
from ac_segmentation.gunpowder.roi import Roi
from ac_segmentation.gunpowder.build import build
from ac_segmentation.gunpowder.nodes.scan import Scan
from ac_segmentation.gunpowder.ext import ZarrFile
from ac_segmentation.gunpowder.batch import Batch
from ac_segmentation.gunpowder.profiling import Timing
from ac_segmentation.gunpowder.array import Array
from ac_segmentation.gunpowder.provider_spec import ProviderSpec
from ac_segmentation.gunpowder.nodes.zarr_source import ZarrSource
from ac_segmentation.gunpowder.nodes.batch_filter import BatchFilter

    

logger = logging.getLogger(__name__)

class TensorStoreSource(ZarrSource):

    def __init__(self, tensorstore=None, array_specs=None, channels_first=True, add_margin=None):
        if array_specs is None:
            self.array_specs = {}
        else:
            self.array_specs = array_specs

        self.channels_first = channels_first
        self.tensorstore = tensorstore
        self.add_margin = add_margin
        self.shape = next(iter(self.tensorstore.values())).shape

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


    def __read_spec(self, array_key, tensorstore):
        dataset = tensorstore

        if array_key in self.array_specs:
            spec = self.array_specs[array_key].copy()
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:
            voxel_size = Coordinate((1,) * len(dataset.shape))
            logger.warning(
                "WARNING: File %s does not contain resolution information for %s, voxel size has been set to %s. This might not be  you want.",
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
                "WARNING: You didn't set 'interpolatable' for %s. Based on the dtype %s, it has been set to %s. This might not be  you want.",
                array_key,
                spec.dtype,
                spec.interpolatable,
            )

        return spec

    def name(self):
        return 'TensorStoreSource[' + list(self.tensorstore.values())[0].kvstore.path + ']'


    def __read(self, data_file, roi):
        c = len(data_file.shape) - self.ndims

        slices = roi.to_slices()

        if self.add_margin:
            slices = tuple(
                    slice(
                        max(0, s.start - self.add_margin) if s.start != 0 else 0,
                        min(self.shape[i], s.stop + self.add_margin),
                        s.step
                    )
                    for i, s in enumerate(slices))


        if self.channels_first:
            array = data_file[(slice(None),) * c + slices].read().result()
        else:
            array = data_file[slices + (slice(None),) * c].read().result()
            array = np.transpose(array, axes=[i + self.ndims for i in range(c)] + list(range(self.ndims)))


        return array


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
                array = self.__read(tensorstore, dataset_roi)
                
                #if self.add_margin:
                    #nshape = array.shape
                    #dataset_roi.shape = nshape
                    #array_spec.roi.shape = nshape
                    #request_spec.roi = array_spec.roi
                    
                # add array to batch
                batch.arrays[array_key] = Array(array, array_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

        
        
class ContrastAdjustWrite(BatchFilter):
    def __init__(self, input_key, output_key, input_arr, output_arr, int_range=None, version='range', mask=None, dsfactor=1, add_margin=None, depth=.9):
        self.input_key = input_key
        self.output_key = output_key
        self.int_range = int_range
        self.version = version
        self.out_array = output_arr
        self.in_array = input_arr
        self.write_objects = []
        self.mask = mask
        self.dsfactor = dsfactor
        self.add_margin=add_margin
        self.depth=.6

    def setup(self):
        pass

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.input_key] = request[self.output_key].copy()
        return deps

    def process(self, batch, request):
        roi = batch.arrays[self.input_key].spec.roi
        slices = roi.to_slices()
        if self.add_margin:
            slices = tuple(
                    slice(
                        max(0, s.start - self.add_margin) if s.start != 0 else 0,
                        min(self.out_array.shape[i], s.stop + self.add_margin),
                        s.step
                    )
                    for i, s in enumerate(slices))

        start = [s.start for s in slices]
        end = [s.stop for s in slices]
        _,_,x1, y1, z1 = start
        _,_,x2, y2, z2 = end
        
        input_data = batch[self.input_key].data
               
        if isinstance(self.mask, np.ndarray):
            ds_start, ds_end = np.ceil(np.array(start) / self.dsfactor).astype(int), np.ceil(np.array(end) / self.dsfactor).astype(int)
            dx1, dy1, dz1 = ds_start[-3:]
            dx2, dy2, dz2 = ds_end[-3:]
            
            mask_slice = self.mask[0,0,dx1:dx2,dy1:dy2,dz1:dz2]
            
            if np.all(mask_slice > 0):
                return  
                     
                                                          

        p1, p2 = np.percentile(input_data, self.int_range)

        
        if np.any(input_data) == True:
            if len(self.in_array.shape) ==5:
                input_data = input_data[0,0,:,:,:]
            scale = 1.0 / (p2 - p1) if p2 > p1 else 1.0
            output_data = np.clip((input_data - p1) * scale, 0, 1)
            output_data = (output_data * 255)                                            
                        

            if len(self.in_array.shape) ==5:
                try:
                    self.write_objects.append([[x1,x2,y1,y2,z1,z2], output_data])
                except:
                    pass
            else:
                self.write_objects.append([[x1,x2,y1,y2,z1,z2], output_data])

    def get_write_objects(self):
        return self.write_objects

    def clear_write_objects(self):
        self.write_objects = []



class ContrastAdjust(BatchFilter):
    def __init__(self, input_key, output_key, int_range=None, version='range'):
        self.input_key = input_key
        self.output_key = output_key
        self.int_range = int_range
        self.version = version

    def setup(self):
        pass

    def prepare(self, request):
        # Ensure the input array is requested
        deps = BatchRequest()
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
        adjusted_array = Array(adjusted_data, spec)

        # Store it in the batch
        batch = Batch()
        batch[self.output_key] = adjusted_array
        
        return batch


class ApplyModel(BatchFilter):
    def __init__(self, model, input_key, ts_array, device, mask=None, dsfactor=1, add_margin=None):
        self.model = model
        self.input_key = input_key
        self.ts_array = ts_array
        self.write_objects = []
        self.device = device
        self.mask = mask
        self.dsfactor = dsfactor
        self.add_margin = add_margin

    def process(self, batch, request):
        # Get the input data
        roi = batch.arrays[self.input_key].spec.roi
        slices = roi.to_slices()
        if self.add_margin:
            slices = tuple(
                    slice(
                        max(0, s.start - self.add_margin) if s.start != 0 else 0,
                        min(self.ts_array.shape[i], s.stop + self.add_margin),
                        s.step
                    )
                    for i, s in enumerate(slices))

        start = [s.start for s in slices]
        end = [s.stop for s in slices]
        _,_,x1, y1, z1 = start
        _,_,x2, y2, z2 = end
        
        if isinstance(self.mask, np.ndarray):
            ds_start, ds_end = np.ceil(np.array(start) / self.dsfactor).astype(int), np.ceil(np.array(end) / self.dsfactor).astype(int)
            dx1, dy1, dz1 = ds_start[-3:]
            dx2, dy2, dz2 = ds_end[-3:]

            if np.all(self.mask[0,0,dx1:dx2,dy1:dy2,dz1:dz2] > 0):
                return
    
        input_data = batch[self.input_key].data
        
        if np.any(input_data):
            tx,ty,tz = tuple((x // 16) * 16 for x in input_data.shape[2:])
            input_data = input_data[:,:,:tx,:ty,:tz]
    
            if input_data.dtype != np.int16:
                input_data = input_data.astype(np.int16)
            if len(input_data.shape)<5:
                x,y,z = input_data.shape
                input_data = input_data.reshape(1, 1, x, y, z)
    
            # Convert input data to a tensor
            input_tensor = torch.from_numpy(input_data).float().to(self.device)
    
            # Run the model
            with torch.inference_mode():
                output_tensor = self.model(input_tensor)
    
            # Convert output tensor to probability map
            output_data = output_tensor[0].data.cpu()
            output_data = torch.special.expit(output_data).numpy()
            
            if np.isneginf(output_data).any():
                output_data[output_data == -np.inf] = 0
                
            self.write_objects.append([[x1,x1+tx,y1,y1+ty,z1,z1+tz], output_data])

        
    def get_write_objects(self):
        return self.write_objects

    def clear_write_objects(self):
        self.write_objects = []
        
 
class Fuse(BatchFilter):
    def __init__(self, input_key, out_arr, x0_adj, y0_adj, z0_adj, flatten):
        self.input_key = input_key
        self.write_objects = []
        self.x0_adj = x0_adj
        self.y0_adj = y0_adj
        self.z0_adj = z0_adj
        self.flatten = flatten
        self.out_arr = out_arr
        self.write_objects = []


    def process(self, batch, request):
        # Get the input data
        roi = batch.arrays[self.input_key].spec.roi
        slices = roi.to_slices()
        _,_,xb, yb, zb = [s.start for s in slices]
        _,_,xb_end, yb_end, zb_end = [s.stop for s in slices]
                
        A_block = batch[self.input_key].data
        
        # ---- SKIP if block is all zeros ----
        if not np.any(A_block):
            return
        
        #adjust according to translation
        out_x0 = self.x0_adj + (xb)
        out_x1 = out_x0 + (xb_end - xb)
        out_y0 = self.y0_adj + (yb)
        out_y1 = out_y0 + (yb_end - yb)
        out_z0 = self.z0_adj + (zb)
        out_z1 = out_z0 + (zb_end - zb)
        
        if self.flatten['surface_map'] is not None:
            smap = flatten['surface_map']   
            if self.flatten['axis'] == 'x':
                #adjust out indices 
                out_x1 += smap.max()
                B_block = np.zeros((1,1,out_x1-out_x0,out_y1-out_y0,out_z1-out_z0))
            
                ix,iy,iz = A_block.shape[-3:]
                for y in range(iy):
                    for z in range(iz):
                        a_row = A_block[:,:,:,y,z]
                        x_shift = smap[y+yb,z+zb]
                        B_block[:, :, x_shift:x_shift+ix, y, z] = a_row
                                                                        

            if self.flatten['axis'] == 'y':
                #adjust out indices 
                out_y1 += smap.max()
                B_block = np.zeros((1,1,out_x1-out_x0,out_y1-out_y0,out_z1-out_z0))
            
                ix,iy,iz = A_block.shape[-3:]
                for x in range(ix):
                    for z in range(iz):
                        a_row = A_block[:,:,x,:,z]
                        y_shift = smap[x+xb,z+zb]
                        B_block[:, :, x, y_shift:y_shift+iy, z] = a_row
                                                                      

            if self.flatten['axis'] == 'z':
                #adjust out indices 
                out_z1 += smap.max()
                B_block = np.zeros((1,1,out_x1-out_x0,out_y1-out_y0,out_z1-out_z0))
            
                ix,iy,iz = A_block.shape[-3:]
                for x in range(ix):
                    for y in range(iy):
                        a_row = A_block[:,:,x,y,:]
                        z_shift = smap[x+xb,y+yb]
                        B_block[:, :, x, y, z_shift:z_shift+iz] = a_row
                                                            
        else:
            B_block = A_block 
            
        self.write_objects.append([[out_x0,out_x1,out_y0,out_y1,out_z0,out_z1], B_block])                                      

    def get_write_objects(self):
        return self.write_objects

    def clear_write_objects(self):
        self.write_objects = []
        
        


def total_volume_shape(arrs, translations):
    mins = []
    maxs = []
    for A,(x,y,z) in zip(arrs, translations):
        X,Y,Z = A.shape[-3:]
        mins.append([x,     y,     z])
        maxs.append([x+X,   y+Y,   z+Z])
    mins = np.min(mins, axis=0)
    maxs = np.max(maxs, axis=0)
    return tuple((maxs - mins).astype(int)), mins


def no_neg(value):
    return value if value >= 0 else 0


def perimeter_weighted_blend(array1, array2, depth=.5):
    weight_map = make_mask(array1.shape[-3:], tuple(int(t*0.5) for t in array1.shape[-3:]), edge=None, bump='zung')
    return (array1 * (1 - weight_map) + array2 * (weight_map))


