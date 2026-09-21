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
    """Gunpowder BatchProvider that reads array blocks directly from in-memory TensorStore datasets instead of a Zarr store on disk.

    Args:
        tensorstore (dict[ArrayKey, tensorstore.TensorStore] | None): Mapping of gunpowder array keys to the TensorStore datasets they should read from.
        array_specs (dict[ArrayKey, ArraySpec] | None): Optional per-key ArraySpec overrides (e.g. voxel size, interpolatable).
        channels_first (bool): Whether the underlying array's leading dimensions are channel dimensions rather than trailing.
        add_margin (int | None): Number of voxels of extra context to read around each requested ROI.

    Example:
        >>> raw = ArrayKey('RAW')  # doctest: +SKIP
        >>> source = TensorStoreSource({raw: input_arr}, {raw: ArraySpec(interpolatable=True)})  # doctest: +SKIP
    """

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
        """Look up the physical offset stored in a dataset's attributes, reversing axis order if the store uses N5-style metadata.

        Args:
            dataset: TensorStore/Zarr dataset whose 'offset' attribute is read.
        """
        if "offset" not in dataset.attrs:
            return None

        if self._rev_metadata():
            return Coordinate(dataset.attrs["offset"][::-1])
        else:
            return Coordinate(dataset.attrs["offset"])

    def _rev_metadata(self):
        """Determine whether this source's underlying Zarr store uses N5 (reversed-axis) metadata.

        Args:
            self (TensorStoreSource): Instance whose `self.store` chunk store is inspected.
        """
        with ZarrFile(self.store, mode="a") as store:
            return isinstance(store.chunk_store, N5Store) or isinstance(store.chunk_store, N5FSStore)

    def setup(self):
        """Register each configured TensorStore array as a provided output of this source.

        Args:
            self (TensorStoreSource): Instance whose `self.tensorstore` mapping is iterated.
        """
        for array_key, tensorstore in self.tensorstore.items():
            spec = self.__read_spec(array_key, tensorstore)
            self.provides(array_key, spec, tensorstore)

    def provides(self, key, spec, tensorstore):
        """Register a new array key as an output provided by this source, storing its spec.

        Args:
            key (ArrayKey): Gunpowder array key being registered.
            spec (ArraySpec): Spec describing the provided array (ROI, voxel size, dtype, etc.).
            tensorstore (tensorstore.TensorStore): TensorStore dataset backing this key, used only for logging its path.
        """
        name = 'TensorStoreSource[' + str(tensorstore.kvstore.path) + ']'
        logger.debug("Current spec of %s:\\n%s", name, self.spec)

        if self.spec is None:
            self._spec = ProviderSpec()

        assert (key not in self.spec), "Node %s is trying to add spec for %s, but is already provided." % (type(self).__name__, key)

        self.spec[key] = copy.deepcopy(spec)
        self.provided_items.append(key)

        logger.debug("%s provides %s with spec %s", name, key, spec)


    def __read_spec(self, array_key, tensorstore):
        """Build (or fill in defaults for) the ArraySpec for a given array key based on its TensorStore dataset's shape and dtype.

        Args:
            array_key (ArrayKey): Gunpowder array key whose spec is being constructed.
            tensorstore (tensorstore.TensorStore): TensorStore dataset used to infer shape, dtype, and default voxel size/ROI.
        """
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
        """Return a human-readable name for this source based on the first TensorStore dataset's path.

        Args:
            self (TensorStoreSource): Instance whose `self.tensorstore` values are inspected.
        """
        return 'TensorStoreSource[' + list(self.tensorstore.values())[0].kvstore.path + ']'


    def __read(self, data_file, roi):
        """Read a sub-array from a TensorStore dataset for the given ROI, optionally padding by `add_margin` and transposing channel axes.

        Args:
            data_file (tensorstore.TensorStore): Dataset to read from.
            roi (Roi): Region of interest, in voxel units, to read.
        """
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
        """Fulfill a gunpowder batch request by reading the requested ROI from each configured TensorStore array and packaging the results into a Batch.

        Args:
            request (BatchRequest): Request specifying which array keys and ROIs to read.
        """
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
    """Gunpowder BatchFilter that percentile-normalizes each incoming block, optionally skips masked blocks, and buffers the results for later assembly into an output tensorstore array.

    Args:
        input_key (ArrayKey): Gunpowder key of the array to read contrast-adjusted data from.
        output_key (ArrayKey): Gunpowder key requested downstream that this filter is responsible for producing.
        input_arr (tensorstore.TensorStore): Source tensorstore array (used to check dimensionality).
        output_arr (tensorstore.TensorStore): Destination tensorstore array the adjusted blocks will eventually be written to.
        int_range (Sequence[float] | None): Percentile range used for contrast normalization.
        version (str): Normalization mode identifier, kept for interface parity.
        mask (np.ndarray | None): Optional downsampled mask array; blocks fully inside the mask are skipped.
        dsfactor (int): Downsample factor used to map block coordinates into the mask's coordinate space.
        add_margin (int | None): Number of voxels of extra context to include around each block's write region.
        depth (float): Unused blending depth parameter retained for interface compatibility (overwritten to 0.6 internally).

    Example:
        >>> contrast = ContrastAdjustWrite(raw, raw, input_arr, output_arr, int_range=[5, 99.5], version='percentile')  # doctest: +SKIP
    """
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
        """No-op setup hook required by the BatchFilter interface.

        Args:
            self (ContrastAdjustWrite): Filter instance.
        """
        pass

    def prepare(self, request):
        """Declare that this filter depends on reading `input_key` for the requested `output_key` ROI.

        Args:
            request (BatchRequest): Downstream request specifying the ROI needed for `output_key`.
        """
        deps = BatchRequest()
        deps[self.input_key] = request[self.output_key].copy()
        return deps

    def process(self, batch, request):
        """Percentile-normalize the input block to 0-255, skip it if fully masked, and append it to the list of pending write objects.

        Args:
            batch (Batch): Batch containing the input array data and its ROI.
            request (BatchRequest): Original batch request (unused directly beyond triggering processing).
        """
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
        x1, y1, z1 = start[-3:]
        x2, y2, z2 = end[-3:]
        
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
        """Return the list of buffered (bounding box, array) writes accumulated by `process`.

        Args:
            self (ContrastAdjustWrite): Filter instance.
        """
        return self.write_objects

    def clear_write_objects(self):
        """Discard all buffered write objects, freeing memory before the next batch.

        Args:
            self (ContrastAdjustWrite): Filter instance.
        """
        self.write_objects = []



class ContrastAdjust(BatchFilter):
    """Gunpowder BatchFilter that applies a min-max or percentile contrast adjustment to a block and returns the result as a new batch, without buffering writes.

    Args:
        input_key (ArrayKey): Gunpowder key of the array to read raw data from.
        output_key (ArrayKey): Gunpowder key the adjusted output will be stored under.
        int_range (Sequence[float] | None): Intensity or percentile range used for normalization.
        version (str): Normalization mode, 'range' for min-max rescaling or 'percentile' for percentile-based rescaling.

    Example:
        >>> contrast = ContrastAdjust(raw, adjusted, int_range=[0, 20000], version='range')  # doctest: +SKIP
    """
    def __init__(self, input_key, output_key, int_range=None, version='range'):
        self.input_key = input_key
        self.output_key = output_key
        self.int_range = int_range
        self.version = version

    def setup(self):
        """No-op setup hook required by the BatchFilter interface.

        Args:
            self (ContrastAdjust): Filter instance.
        """
        pass

    def prepare(self, request):
        """Declare that this filter depends on reading `input_key` for the requested `output_key` ROI.

        Args:
            request (BatchRequest): Downstream request specifying the ROI needed for `output_key`.
        """
        # Ensure the input array is requested
        deps = BatchRequest()
        deps[self.input_key] = request[self.output_key].copy()
        return deps

    def process(self, batch, request):
        """Apply min-max or percentile contrast rescaling to the input block and return it as a new batch under `output_key`.

        Args:
            batch (Batch): Batch containing the input array data to adjust.
            request (BatchRequest): Downstream request specifying the ROI for `output_key`.
        """
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
    """Gunpowder BatchFilter that runs a trained segmentation model on each incoming block and buffers the resulting probability maps for later assembly into an output tensorstore array.

    Args:
        model (torch.nn.Module): Trained segmentation model to run on each block.
        input_key (ArrayKey): Gunpowder key of the array to read raw data from.
        ts_array (tensorstore.TensorStore): Output tensorstore array the model's predictions will eventually be written to.
        device (torch.device): Device to run model inference on.
        mask (np.ndarray | None): Optional downsampled mask array; blocks fully inside the mask are skipped.
        dsfactor (int): Downsample factor used to map block coordinates into the mask's coordinate space.
        add_margin (int | None): Number of voxels of extra context to include around each block's write region.

    Example:
        >>> apply_model = ApplyModel(model, raw, output_arr, device=torch.device('cuda:0'))  # doctest: +SKIP
    """
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
        """Run the model on the input block (after mask/skip checks and shape trimming to a multiple of 16), convert the output logits to a probability map, and buffer it as a pending write object.

        Args:
            batch (Batch): Batch containing the input array data to run inference on.
            request (BatchRequest): Original batch request (unused directly beyond triggering processing).
        """
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
        x1, y1, z1 = start[-3:]
        x2, y2, z2 = end[-3:]
        
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
        """Return the list of buffered (bounding box, array) writes accumulated by `process`.

        Args:
            self (ApplyModel): Filter instance.
        """
        return self.write_objects

    def clear_write_objects(self):
        """Discard all buffered write objects, freeing memory before the next batch.

        Args:
            self (ApplyModel): Filter instance.
        """
        self.write_objects = []
        
 
class Fuse(BatchFilter):
    """Gunpowder BatchFilter that places each incoming block into the coordinate frame of a larger fused output volume, optionally remapping voxels along a flattening surface map, and buffers the result as a pending write object.

    Args:
        input_key (ArrayKey): Gunpowder key of the array to read block data from.
        out_arr (tensorstore.TensorStore): Output tensorstore array this block will eventually be written into.
        x0_adj (int): X offset added to place this array's local coordinates into the fused volume.
        y0_adj (int): Y offset added to place this array's local coordinates into the fused volume.
        z0_adj (int): Z offset added to place this array's local coordinates into the fused volume.
        flatten (dict): Optional surface-flattening config with keys 'surface_map' (np.ndarray | None) and 'axis' ('x', 'y', or 'z') describing how to remap voxels along a warped surface.

    Example:
        >>> fuse = Fuse(raw, out_arr, x0_adj=0, y0_adj=0, z0_adj=0, flatten={'surface_map': None, 'axis': 'x'})  # doctest: +SKIP
    """
    def __init__(self, input_key, out_arr, x0_adj, y0_adj, z0_adj, flatten):
        self.input_key = input_key
        self.write_objects = []
        self.x0_adj = x0_adj
        self.y0_adj = y0_adj
        self.z0_adj = z0_adj
        self.flatten = flatten
        self.out_arr = out_arr
        self.is_5d = out_arr.ndim == 5

    def process(self, batch, request):
        """Translate the input block into the fused output volume's coordinate frame, remapping voxels along the configured flattening surface if provided, and buffer it as a pending write object.

        Args:
            batch (Batch): Batch containing the input array data and its ROI.
            request (BatchRequest): Original batch request (unused directly beyond triggering processing).
        """
        roi = batch.arrays[self.input_key].spec.roi
        slices = roi.to_slices()

        # Unpack only the last 3 spatial dims regardless of total ndim
        xb,  yb,  zb     = [s.start for s in slices[-3:]]
        xb_end, yb_end, zb_end = [s.stop  for s in slices[-3:]]

        A_block = batch[self.input_key].data

        if not np.any(A_block):
            return

        out_x0 = self.x0_adj + xb
        out_x1 = out_x0 + (xb_end - xb)
        out_y0 = self.y0_adj + yb
        out_y1 = out_y0 + (yb_end - yb)
        out_z0 = self.z0_adj + zb
        out_z1 = out_z0 + (zb_end - zb)

        if self.flatten['surface_map'] is not None:
            smap = self.flatten['surface_map']  # was wrongly using bare `flatten`

            ix, iy, iz = A_block.shape[-3:]

            if self.flatten['axis'] == 'x':
                out_x1 += smap.max()
                spatial = (out_x1-out_x0, out_y1-out_y0, out_z1-out_z0)
                B_block = np.zeros(((1,1)+spatial) if self.is_5d else spatial, dtype=A_block.dtype)
                for y in range(iy):
                    for z in range(iz):
                        a_row = A_block[..., :, y, z]   # [...] handles both 3D and 5D
                        x_shift = smap[y+yb, z+zb]
                        if self.is_5d:
                            B_block[:, :, x_shift:x_shift+ix, y, z] = a_row
                        else:
                            B_block[x_shift:x_shift+ix, y, z] = a_row

            elif self.flatten['axis'] == 'y':
                out_y1 += smap.max()
                spatial = (out_x1-out_x0, out_y1-out_y0, out_z1-out_z0)
                B_block = np.zeros(((1,1)+spatial) if self.is_5d else spatial, dtype=A_block.dtype)
                for x in range(ix):
                    for z in range(iz):
                        a_row = A_block[..., x, :, z]
                        y_shift = smap[x+xb, z+zb]
                        if self.is_5d:
                            B_block[:, :, x, y_shift:y_shift+iy, z] = a_row
                        else:
                            B_block[x, y_shift:y_shift+iy, z] = a_row

            elif self.flatten['axis'] == 'z':
                out_z1 += smap.max()
                spatial = (out_x1-out_x0, out_y1-out_y0, out_z1-out_z0)
                B_block = np.zeros(((1,1)+spatial) if self.is_5d else spatial, dtype=A_block.dtype)
                for x in range(ix):
                    for y in range(iy):
                        a_row = A_block[..., x, y, :]
                        z_shift = smap[x+xb, y+yb]
                        if self.is_5d:
                            B_block[:, :, x, y, z_shift:z_shift+iz] = a_row
                        else:
                            B_block[x, y, z_shift:z_shift+iz] = a_row
        else:
            B_block = A_block

        self.write_objects.append([[out_x0, out_x1, out_y0, out_y1, out_z0, out_z1], B_block])

    def get_write_objects(self):
        """Return the list of buffered (bounding box, array) writes accumulated by `process`.

        Args:
            self (Fuse): Filter instance.
        """
        return self.write_objects

    def clear_write_objects(self):
        """Discard all buffered write objects, freeing memory before the next batch.

        Args:
            self (Fuse): Filter instance.
        """
        self.write_objects = []
        
        
class VoxelRelabel(BatchFilter):
    """Gunpowder BatchFilter that, for blocks overlapping known skeletons, thresholds and labels connected components and relabels each one to the ID of its nearest skeleton, buffering the result for later writing.

    Args:
        input_key (ArrayKey): Gunpowder key of the array to read raw/probability data from.
        output_key (ArrayKey): Gunpowder key requested downstream that this filter is responsible for producing.
        input_arr (tensorstore.TensorStore): Source tensorstore array (used to check dimensionality).
        output_arr (tensorstore.TensorStore): Destination tensorstore array the relabeled blocks will eventually be written to.
        skels (list[cloudvolume.Skeleton]): Skeletons used to assign nearest-skeleton IDs to labeled voxels.

    Example:
        >>> relabel = VoxelRelabel(raw, raw, input_arr, output_arr, skels)  # doctest: +SKIP
    """
    def __init__(self, input_key, output_key, input_arr, output_arr, skels):
        self.input_key = input_key
        self.output_key = output_key
        self.out_array = output_arr
        self.in_array = input_arr
        self.write_objects = []
        self.skels = skels
        self.is_5d = input_arr.ndim == 5

    def setup(self):
        """No-op setup hook required by the BatchFilter interface.

        Args:
            self (VoxelRelabel): Filter instance.
        """
        pass

    def prepare(self, request):
        """Declare that this filter depends on reading `input_key` for the requested `output_key` ROI.

        Args:
            request (BatchRequest): Downstream request specifying the ROI needed for `output_key`.
        """
        deps = BatchRequest()
        deps[self.input_key] = request[self.output_key].copy()
        return deps

    def process(self, batch, request):
        """For blocks that overlap any of `self.skels`, threshold and label connected components in the input block and relabel each voxel to its nearest skeleton's ID, buffering the result as a pending write object.

        Args:
            batch (Batch): Batch containing the input array data and its ROI.
            request (BatchRequest): Original batch request (unused directly beyond triggering processing).
        """
        roi = batch.arrays[self.input_key].spec.roi
        slices = roi.to_slices()

        # Unpack only spatial dims from the end
        x1, y1, z1 = [s.start for s in slices[-3:]]
        x2, y2, z2 = [s.stop  for s in slices[-3:]]

        input_data = batch[self.input_key].data

        if len(filter_skeletons(self.skels, [x1, x2, y1, y2, z1, z2])) > 0:
            if np.any(input_data):
                # Always extract the 3D spatial block the same way
                input_data = input_data[0, 0] if self.is_5d else input_data

                input_data = input_data.astype('uint64')
                binary_arr = threshold_binarize_array(input_data, threshold=15)
                labeled_vol, num_feat = label_binary_array(binary_arr, size_threshold=10)
                output_data = relabel_volume_by_nearest_skeleton(
                    labeled_vol=labeled_vol, skeletons=self.skels, offset=(x1, y1, z1)
                )

                try:
                    self.write_objects.append([[x1, x2, y1, y2, z1, z2], output_data])
                except:
                    pass

    def get_write_objects(self):
        """Return the list of buffered (bounding box, array) writes accumulated by `process`.

        Args:
            self (VoxelRelabel): Filter instance.
        """
        return self.write_objects

    def clear_write_objects(self):
        """Discard all buffered write objects, freeing memory before the next batch.

        Args:
            self (VoxelRelabel): Filter instance.
        """
        self.write_objects = []



def total_volume_shape(arrs, translations):
    """Compute the shape and minimum corner of the bounding box that contains all given arrays once placed at their respective translations.

    Args:
        arrs (list[np.ndarray | tensorstore.TensorStore]): Arrays whose spatial extents are combined.
        translations (list[tuple[int, int, int]]): (x, y, z) placement offset for each array in arrs, aligned by index.

    Example:
        >>> shape, mins = total_volume_shape([arr1, arr2], [(0, 0, 0), (100, 0, 0)])  # doctest: +SKIP
    """
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
    """Clamp a value to be non-negative.

    Args:
        value (int | float): Value to clamp.

    Example:
        >>> no_neg(-5)
        0
    """
    return value if value >= 0 else 0


def perimeter_weighted_blend(array1, array2, depth=.5):
    """Blend two overlapping arrays using a perimeter-based weight mask so the second array's edges fade smoothly into the first.

    Args:
        array1 (np.ndarray): Base array (e.g. existing output data).
        array2 (np.ndarray): New array to blend in (e.g. freshly written block).
        depth (float): Unused directly here; retained for interface compatibility with the underlying bump-mask blending.

    Example:
        >>> blended = perimeter_weighted_blend(existing_block, new_block)  # doctest: +SKIP
    """
    weight_map = make_mask(array1.shape[-3:], tuple(int(t*0.5) for t in array1.shape[-3:]), edge=None, bump='zung')
    return (array1 * (1 - weight_map) + array2 * (weight_map))
    
 
###relabel function  
    
def label_binary_array(binary_arr, size_threshold=20):
    """Label connected components in a binary array and drop components smaller than a size threshold.

    Args:
        binary_arr (np.ndarray): Boolean or 0/1 array whose foreground voxels will be connected-component labeled.
        size_threshold (int): Minimum connected-component size to keep.

    Example:
        >>> labeled, num_labels = label_binary_array(binary_arr, size_threshold=10)  # doctest: +SKIP
    """
    
    labeled_arr, num_features = cc3d.connected_components(binary_arr, connectivity=6, return_N=True)
    if num_features > 1:
        labeled_arr = skimage.morphology.remove_small_objects(
            labeled_arr, min_size=size_threshold, connectivity=3, out=labeled_arr)
        
    return labeled_arr, len(np.unique(labeled_arr))

def threshold_binarize_array(arr, threshold=0.2):
    """Convert a probability/intensity array to a boolean mask by thresholding.

    Args:
        arr (np.ndarray): Input array of probability or intensity values.
        threshold (float): Value at or above which a voxel is treated as foreground.

    Example:
        >>> mask = threshold_binarize_array(prob_arr, threshold=15)  # doctest: +SKIP
    """
    return (arr >= threshold)


def relabel_volume_by_nearest_skeleton(labeled_vol, skeletons, offset=(0, 0, 0)):
    """Relabel each connected component in a labeled volume to the ID of the nearest skeleton vertex, restricted to components actually touched by a skeleton.

    Args:
        labeled_vol (np.ndarray): Connected-component labeled volume to relabel.
        skeletons (list[cloudvolume.Skeleton]): Skeletons whose vertices are used as relabeling targets.
        offset (tuple[int, int, int]): (x, y, z) offset mapping labeled_vol's local voxel coordinates into the skeletons' global coordinate space.

    Example:
        >>> relabeled = relabel_volume_by_nearest_skeleton(labeled_vol, skels, offset=(x1, y1, z1))  # doctest: +SKIP
    """
    pts = []
    ids = []
    for sk in skeletons:
        verts = list(sk.vertices)
        pts += verts
        ids += [sk.id] * len(verts)
    pts = np.asarray(pts, dtype=np.float64)  
    ids = np.asarray(ids)                   

    offset = np.asarray(offset, dtype=np.float64)

    # find which CC labels are hit by a skeleton vertex
    pts_relative = np.round(pts - offset).astype(int)
    shape = np.array(labeled_vol.shape)
    valid = np.all((pts_relative >= 0) & (pts_relative < shape), axis=1)
    pts_rel_valid = pts_relative[valid]
    xi_s, yi_s, zi_s = pts_rel_valid[:, 0], pts_rel_valid[:, 1], pts_rel_valid[:, 2]
    hit_labels = set(labeled_vol[xi_s, yi_s, zi_s].tolist())
    hit_labels.discard(0)

    all_labels = set(np.unique(labeled_vol).tolist())
    all_labels.discard(0)

    # only process voxels in hit components
    xi, yi, zi = np.nonzero(labeled_vol > 0)
    cc_labels = labeled_vol[xi, yi, zi]
    in_hit_component = np.isin(cc_labels, list(hit_labels))
    xi, yi, zi = xi[in_hit_component], yi[in_hit_component], zi[in_hit_component]

    vox_absolute = np.column_stack([xi, yi, zi]) + offset

    tree = cKDTree(pts)
    dist, idx = tree.query(vox_absolute)

    out = np.zeros_like(labeled_vol, dtype='uint64')
    out[xi, yi, zi] = ids[idx]
    return out
            
    
    
def filter_skeletons(skels, bbox): #x1,x2,y1,y2,z1,z2
    """Return the subset of skeletons that have at least one vertex within a given bounding box.

    Args:
        skels (list[cloudvolume.Skeleton]): Skeletons to filter.
        bbox (tuple[int, int, int, int, int, int]): (x1, x2, y1, y2, z1, z2) bounding box in the skeletons' coordinate space.

    Example:
        >>> nearby = filter_skeletons(all_skels, [0, 100, 0, 100, 0, 100])  # doctest: +SKIP
    """
    x1,x2,y1,y2,z1,z2 = bbox
    lower = np.array([x1, y1, z1])
    upper = np.array([x2, y2, z2])
    
    matches = []
    for sk in skels:
        verts = np.array(sk.vertices, dtype=int)  # shape (N, 3)
        in_bounds = np.all((verts >= lower) & (verts <= upper), axis=1)
        if in_bounds.any():
            matches.append(sk)
    return matches
    
    
def no_neg(value):
    """Clamp a value to be non-negative.

    Args:
        value (int | float): Value to clamp.

    Example:
        >>> no_neg(-5)
        0
    """
    return value if value >= 0 else 0

