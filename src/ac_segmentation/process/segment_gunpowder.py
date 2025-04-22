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

import tensorstore as ts
from ac_segmentation.utils.tensorstore import open_tensor, create_tensor
from ac_segmentation.utils.preprocess import lut_preprocess_array
import ac_segmentation.neurotorch.nets.RSUNet
RSUNet = ac_segmentation.neurotorch.nets.RSUNet.RSUNet


logger = logging.getLogger(__name__)

class TensorStoreSource(gp.ZarrSource):

    def __init__( 
        self,
        tensorstore=None,
        array_specs=None,
        channels_first=True
    ):


        if array_specs is None:
            self.array_specs = {}
        else:
            self.array_specs = array_specs

        self.channels_first = channels_first
        self.tensorstore = tensorstore


    ##NEEDS TO BE EDITED
    def _get_offset(self, dataset):
        if "offset" not in dataset.attrs:
            return None

        if self._rev_metadata():
            return Coordinate(dataset.attrs["offset"][::-1])
        else:
            return Coordinate(dataset.attrs["offset"])


    ##NEEDS TO BE EDITED
    def _rev_metadata(self):
        with ZarrFile(self.store, mode="a") as store:
            return isinstance(store.chunk_store, N5Store) or isinstance(
                store.chunk_store, N5FSStore
            )

    def setup(self):
        for array_key, tensorstore in self.tensorstore.items():
            spec = self.__read_spec(array_key, tensorstore)
            self.provides(array_key, spec, tensorstore)

    def provides(self, key, spec, tensorstore):
        """Introduce a new output provided by this :class:`BatchProvider`.

        Implementations should call this in their :func:`setup` method, which
        will be called when the pipeline is build.

        Args:

            key (:class:`ArrayKey` or :class:`GraphKey`):

                The array or point set key provided.

            spec (:class:`ArraySpec` or :class:`GraphSpec`):

                The spec of the array or point set provided.
        """

        name = 'TensorStoreSource[' + str(tensorstore.kvstore.path) + ']'
        logger.debug("Current spec of %s:\n%s", name, self.spec)

        if self.spec is None:
            self._spec = ProviderSpec()

        assert (
            key not in self.spec
        ), "Node %s is trying to add spec for %s, but is already " "provided." % (
            type(self).__name__,
            key,
        )

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
                batch.arrays[array_key] = Array(
                    self.__read(tensorstore, dataset_roi),
                    array_spec,
                )

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
            array = np.transpose(
                array, axes=[i + self.ndims for i in range(c)] + list(range(self.ndims))
            )

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
                    "WARNING: File %s does not contain resolution information "
                    "for %s, voxel size has been set to %s. This "
                    "might not be what you want.",
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
                "dtype %s provided in array_specs for %s, "
                "but differs from dataset dtype %s"
                % (self.array_specs[array_key].dtype, array_key, dataset.dtype.name)
            )
        else:
            spec.dtype = dataset.dtype.name

        if spec.interpolatable is None:
            spec.interpolatable = np.issubdtype(spec.dtype, np.floating) or (
                spec.dtype == np.uint8
            )
            logger.warning(
                "WARNING: You didn't set 'interpolatable' for %s "
                ". Based on the dtype %s, it has been "
                "set to %s. This might not be what you want.",
                array_key,
                spec.dtype,
                spec.interpolatable,
            )

            
        return spec


    def name(self):
        return 'TensorStoreSource[' + list(self.tensorstore.values())[0].kvstore.path + ']'


class ContrastAdjustment(gp.BatchFilter):
    def __init__(self, input_key, output_key, int_range=None, version='range'):
        self.input_key = input_key
        self.output_key = output_key
        self.int_range = int_range
        self.version = version

    def setup(self):
        # Specify the output array (contrast-adjusted data)
        #self.provides(self.output_key, self.spec[self.input_key].copy())
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
                adjusted_data = lut_preprocess_array(input_data, r1, r2)

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
        del self.write_objects


import itertools
def create_chunked_dims(arr_shape, chunk_size):
    # Ensure chunk_size is appropriate for the arr_shape length
    if len(arr_shape) != len(chunk_size):
        raise ValueError("arr_shape and chunk_size must have the same number of dimensions")

    # Get indexing combinations
    start_indices = []
    end_indices = []

    for dim_size, chunk in zip(arr_shape, chunk_size):
        # Generate the start indices for each chunk
        start_indices.append(list(range(0, dim_size, chunk)))
        # Generate the end indices for each chunk, ensuring not to exceed the array size
        end_indices.append([min(dim_size, start + chunk) for start in start_indices[-1]])

    # Create all combinations of start and end indices across all dimensions
    start = [list(item) for item in itertools.product(*start_indices)]
    end = [list(item) for item in itertools.product(*end_indices)]

    return start, end


def segment_gunpowder(input_path, output_path, checkpoint, iter_size=(64,64,64), stride=(32,32,32), batch_size=5, gpu_device=None, cpus=20, preprocess={'method':'percentile','values':[96,97]}):
    
    if int(cpus) > int(os.cpu_count()):
        cpus = os.cpu_count()
    torch.set_num_threads(cpus) 
    
    # Set-up model
    device = torch.device("cuda:{}".format(gpu_device) if gpu_device is not None else "cpu")
    model = RSUNet()
    model = model.to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    
    #get input array size
    if isinstance(input_path, str):
        input_arr = open_tensor(input_path, bytes_limit= 100_000_000, driver='zarr')
    else:
        input_arr = input_path

    start_req = (0,0,0)
    if len(input_arr.shape) == 5:
        iter_size = (1,1,) + iter_size
        stride = (0,0,) + stride
        start_req = (0,0,0,0,0)
        
    chunk_size = batch_size*np.array(iter_size)
    
    #define the pipeline
    raw = gp.ArrayKey('RAW')
    source = TensorStoreSource(
        {
            raw: input_arr
        },
        {
            raw: gp.ArraySpec(interpolatable=True)
        })
    
    #create tensorstore
    if isinstance(output_path, str):
        if os.path.isdir(output_path):
            output_arr = open_tensor(output_path, bytes_limit= 100_000_000, driver='zarr3')
        else:
            output_arr = create_tensor(fpath=output_path, arr_shape=input_arr.shape[-3:], dtype = 'float32', fill_value=-np.inf, driver='zarr3')
    else:
        output_arr = output_path

    
    # Define the chunk size
    iter_coord = gp.Coordinate(iter_size) 
    
    # Define the scan request with overlap
    scan_request = gp.BatchRequest()
    scan_request[raw] = gp.Roi(start_req, iter_coord)

    # Define the overlap
    chunk_overlap = gp.Coordinate(stride)

    # Create the Scan node
    scan = gp.Scan(scan_request)
    
    # Create the ApplyModel instance
    apply_model = ApplyModel(model, raw, output_arr, device)
    
    # Build the pipeline with Scan
    method, values = preprocess['method'], preprocess['values']
    pipeline = (
        source +
        ContrastAdjustment(raw,raw,values,version=method) +
        apply_model +
        scan
    )
    
    stime = datetime.now()
    start_prime,end_prime = create_chunked_dims(arr_shape=input_arr.shape, chunk_size=chunk_size)
    start_overlap,end_overlap = [tuple(np.array(x)+np.array(stride)) for x in start_prime], [tuple(np.array(x)+np.array(stride)) for x in end_prime]
    

    for i in range(len(end_prime)):
        end_prime[i] = np.minimum(end_prime[i], np.array(input_arr.shape))
    for i in range(len(end_overlap)):
        end_overlap[i] = np.minimum(end_overlap[i], np.array(input_arr.shape))

    
    save = np.array(start_prime[0])
    for i in range(len(start_prime)):
        arr1,arr2 = np.array(start_prime[i]),np.array(end_prime[i])
        if np.any(arr2-arr1 < np.array(iter_size)) == True:
            start_prime[i] = np.minimum(arr1,save)
        else:
            save = arr1

        arr1,arr2 = np.array(start_overlap[i]),np.array(end_overlap[i])
        if np.any(arr2-arr1 < np.array(iter_size)) == True:
            start_overlap[i] = np.minimum(arr1,save)
        else:
            save = arr1

    start,end = start_prime+start_overlap, end_prime+end_overlap
    with gp.build(pipeline):
        for i in range(len(start)):
            end[i] = np.minimum(end[i], np.array(input_arr.shape))
            arr = np.array(end[i])-np.array(start[i])
            total_roi = gp.Roi(start[i], arr)

            iter_new = np.minimum(arr, np.array(iter_size))
            scan_request[raw] = gp.Roi(start_req, iter_coord)
        
            # Create a request for the entire volume
            request = gp.BatchRequest()
            request[raw] = total_roi

            # Request the batch
            batch = pipeline.request_batch(request)
    
    # Retrieve the list of write objects and write them
    write_objects = apply_model.get_write_objects()
    for write in write_objects:
        write.result()
    apply_model.clear_write_objects()
    etime = datetime.now()
    print(etime-stime)
    
    #output seg card
    compute = {'GPUs': gpu_device} if gpu_device else {'CPUs': cpus}
    seg_card = {'date':datetime.today().strftime('%Y-%m-%d'),
                'paths':{'inpath':input_path,'outpath':output_path}, 
                'preprocessing':{'method':method,'values':values}, 
                'compute':compute,
                'time_lapse':etime-stime}

    with open(os.path.join(output_path, "seg_card.txt"), 'w') as f:
        f.write(str(seg_card))
    
    return seg_card
