import argschema
import numpy as np
from scipy.spatial import cKDTree
import cc3d
import skimage
from datetime import datetime

from ac_segmentation.utils.tensorstore import open_tensor, AWS_Parameters, create_kvstore, create_tensor
from ac_segmentation.utils.io import write_cv_skels_tar
from ac_segmentation.utils.h5_skeletons import *
from ac_segmentation.utils.h5_reconnect import *
from ac_segmentation.utils.preprocess import create_chunked_dims, create_overlap_chunks
from ac_segmentation.methods.nodes import VoxelRelabel, TensorStoreSource, filter_skeletons

import multiprocessing as mp
mp.set_start_method('forkserver', force=True)

from ac_segmentation.gunpowder.array_spec import ArraySpec
from ac_segmentation.gunpowder.array import ArrayKey
from ac_segmentation.gunpowder.coordinate import Coordinate
from ac_segmentation.gunpowder.batch_request import BatchRequest
from ac_segmentation.gunpowder.roi import Roi
from ac_segmentation.gunpowder.build import build
from ac_segmentation.gunpowder.nodes.scan import Scan


def voxel_relabel_gunpowder(input_arr, output_arr, skel_path, iter_size=(64,64,64), batch_size=3, cutout=None):
    """Relabel connected components in an input volume to their nearest known skeleton's ID and write the result to an output tensorstore array.

    Builds a gunpowder pipeline that scans the input volume in overlapping blocks,
    queries a skeleton store for skeletons near each block, and for blocks that
    overlap at least one skeleton, thresholds/labels the block and relabels each
    component to the ID of its nearest skeleton before writing it out.

    Args:
        input_arr (tensorstore.TensorStore): Input labeled/probability volume to relabel.
        output_arr (tensorstore.TensorStore): Output tensorstore array where relabeled voxels are written.
        skel_path (str): Path to the skeleton store queried for nearby skeletons.
        iter_size (tuple[int, int, int]): Spatial size of each scan block before batching.
        batch_size (int): Number of blocks per gunpowder scan chunk.
        cutout (Sequence[int] | None): Optional (x1, x2, y1, y2, z1, z2) bounding box restricting which blocks are processed and queried.

    Example:
        >>> input_arr = open_tensor('s3://bucket/probs.zarr')  # doctest: +SKIP
        >>> output_arr = create_tensor('s3://bucket/labels.zarr', arr_shape=input_arr.shape, dtype='uint64')  # doctest: +SKIP
        >>> voxel_relabel_gunpowder(input_arr, output_arr, skel_path='s3://bucket/skeletons')  # doctest: +SKIP
    """

    is_5d = input_arr.ndim == 5

    raw = ArrayKey('RAW')
    source = TensorStoreSource(
        {raw: input_arr},
        {raw: ArraySpec(interpolatable=True)}
    )

    if is_5d:
        iter_size = (1, 1) + tuple(iter_size)
        start_req = (0, 0, 0, 0, 0)
    else:
        iter_size = tuple(iter_size)
        start_req = (0, 0, 0)

    chunk_size = batch_size * np.array(iter_size)
    iter_coord = Coordinate(iter_size)

    scan_request = BatchRequest()
    scan_request[raw] = Roi(start_req, iter_coord)
    scan = Scan(scan_request, num_workers=0)

    start, end = create_chunked_dims(arr_shape=input_arr.shape, chunk_size=chunk_size)

    if cutout:
        start_new, end_new = [], []
        x1, x2, y1, y2, z1, z2 = cutout
        for i, (s, e) in enumerate(zip(start, end)):
            offset = np.array([x1, y1, z1])
            s, e = np.array(s[-3:]) + offset, np.array(e[-3:]) + offset
            if (s[0] <= x2 and s[1] <= y2 and s[2] <= z2):
                e[-3:] = np.minimum(np.array([x2, y2, z2]) + (iter_size[-1] / 2), np.array(e[-3:]))
                if is_5d:
                    s = np.concatenate(([0, 0], s))
                    e = np.concatenate(([1, 1], e))
                start_new.append(s)
                end_new.append(e)
        start, end = start_new, end_new
    else:
        # If no cutout, define a broad bounding box for skeleton query
        x1, y1, z1 = 0, 0, 0
        x2, y2, z2 = input_arr.shape[-3:]

    for i in range(len(start)):
        start[i], end[i] = np.minimum(start[i], np.array(input_arr.shape)), np.minimum(end[i], np.array(input_arr.shape))
        dif = np.array(end[i]) - np.array(start[i])
        if np.any(dif[-3:] < iter_size[-1]):
            bx, by, bz = np.array(end[i][-3:]) - np.array(iter_size[-3:])
            start[i][-3:] = [bx, by, bz]

    if len(start) == 0:
        print('Batch_size needs to be lowered to accommodate the cutout size.')
        return

    stime = datetime.now()
    all_skels, shards = query_skeletons_by_bb([x1, y1, z1, x2, y2, z2], skel_path, n_workers=10)

    for ind, i in enumerate(range(len(start))):

        sx1, sy1, sz1 = start[i][-3:]
        sx2, sy2, sz2 = end[i][-3:]

        skels = filter_skeletons(all_skels, [sx1, sx2, sy1, sy2, sz1, sz2])

        if len(skels) > 0:
            print(start[i], end[i], "# Skels: ", len(skels))

            relabel = VoxelRelabel(raw, raw, input_arr, output_arr, skels)

            pipeline = (source + relabel + scan)

            with build(pipeline):
                arr = np.array(end[i]) - np.array(start[i])
                total_roi = Roi(start[i], arr)

                request = BatchRequest()
                request[raw] = total_roi
                batch = pipeline.request_batch(request)

                write_objects = relabel.get_write_objects()
                if len(write_objects) > 0:
                    indices = [item[0] for item in write_objects]
                    x1r, x2r, y1r, y2r, z1r, z2r = [max(slot) for slot in zip(*indices)]
                    mx1, mx2, my1, my2, mz1, mz2  = [min(slot) for slot in zip(*indices)]

                    success = False
                    max_retries = 10
                    for attempt in range(max_retries):
                        try:
                            if is_5d:
                                temp_arr = output_arr[:, :, mx1:x2r, my1:y2r, mz1:z2r].read().result()
                                for write in write_objects:
                                    ox1, ox2, oy1, oy2, oz1, oz2 = write[0]
                                    arr2 = temp_arr[0, 0, ox1-mx1:ox2-mx1, oy1-my1:oy2-my1, oz1-mz1:oz2-mz1]
                                    write_data = np.maximum(arr2, write[1])
                                    temp_arr[:, :, ox1-mx1:ox2-mx1, oy1-my1:oy2-my1, oz1-mz1:oz2-mz1] = write_data[None, None, :]
                                if np.any(temp_arr):
                                    output_arr[:, :, mx1:x2r, my1:y2r, mz1:z2r].write(temp_arr).result()
                            else:
                                temp_arr = output_arr[mx1:x2r, my1:y2r, mz1:z2r].read().result()
                                for write in write_objects:
                                    ox1, ox2, oy1, oy2, oz1, oz2 = write[0]
                                    arr2 = temp_arr[ox1-mx1:ox2-mx1, oy1-my1:oy2-my1, oz1-mz1:oz2-mz1]
                                    write_data = np.maximum(arr2, write[1])
                                    temp_arr[ox1-mx1:ox2-mx1, oy1-my1:oy2-my1, oz1-mz1:oz2-mz1] = write_data
                                if np.any(temp_arr):
                                    output_arr[mx1:x2r, my1:y2r, mz1:z2r].write(temp_arr).result()

                            success = True
                            break
                        except Exception as e:
                            print(f"Attempt {attempt+1} failed: {e}")
                            if attempt < max_retries - 1:
                                time.sleep(5)

                    relabel.clear_write_objects()
                    etime = datetime.now()
                    print(i, etime - stime)
        else:
            print(start[i], end[i], 'no skeletons')

                                                                                
class VoxelRelabelParameters(argschema.ArgSchema):
    """Argschema parameter schema defining the input/output paths, skeleton store path, cutout, and S3 access options for the voxel relabeling run.

    Args:
        input_path (str): Path (local or s3://) to the input labeled/probability volume.
        skel_path (str): Path to the skeleton store queried for nearby skeletons.
        output_path (str): Path (local or s3://) where the relabeled output volume is written.
        cutout (Any | None): Optional bounding box (as a comma-separated string or list) restricting processing to a sub-region.
        region (str): AWS region used for S3 access.
        endpoint (str | None): Custom S3-compatible endpoint URL.
        profile (str | None): Named AWS credentials profile to use for S3 access.
    """
    input_path = argschema.fields.String(required=True)
    skel_path = argschema.fields.String(required=True)
    output_path = argschema.fields.String(required=True)
    cutout = argschema.fields.Raw(required=False, allow_none=True, missing=None)

    region = argschema.fields.String(required=False, default='us-west-2')
    endpoint = argschema.fields.String(required=False, default=None, allow_none=True)
    profile = argschema.fields.String(required=False, default=None, allow_none=True)
    

class VoxelRelabelModule(argschema.ArgSchemaParser):
    """Argschema module that loads an input labeled volume, relabels it against a skeleton store via `voxel_relabel_gunpowder`, and writes the result to an output tensorstore array.

    Args:
        None: This class carries no constructor args beyond argschema's ArgSchemaParser.
    """
    default_schema = VoxelRelabelParameters
       

    def run(self):
        """Parse run parameters, open the input/output tensorstore arrays, and run the voxel-relabeling pipeline.

        Converts "None" string args to actual None, parses `cutout` from a string into a
        list if needed, opens the input volume (from a local path or S3), creates or
        opens the output tensorstore array, and runs `voxel_relabel_gunpowder` to produce
        the relabeled volume.

        Args:
            self (VoxelRelabelModule): Instance whose self.args holds the run configuration.

        Example:
            >>> mod = VoxelRelabelModule(input_data={  # doctest: +SKIP
            ...     "input_path": "s3://bucket/probs.zarr",
            ...     "skel_path": "s3://bucket/skeletons",
            ...     "output_path": "s3://bucket/labels.zarr",
            ... }, args=[])
            >>> mod.run()  # doctest: +SKIP
        """
        for key, value in self.args.items():
            if value == 'None':
                self.args[key] = None
                
        # --- Convert bound_box from string to list if present ---
        if self.args['cutout'] and type(self.args['cutout'])==str:
            self.args['cutout'] = [int(x.strip("'")) for x in self.args["cutout"].split(',')]
                                                     
             
        kvstore_in, kvstore_out = None, None                                   
        in_path = self.args['input_path']
        out_path = self.args['output_path']
        
        
        if not self.args['endpoint']:
            endpoint=None
            
        if 's3://' in self.args['input_path']:
            AWS_param = AWS_Parameters(profile=self.args['profile'], region=self.args['region'], endpoint_url=endpoint)      
            kvstore_in = create_kvstore(fpath=str(in_path), store='s3', AWS_param=AWS_param)                                              
        
        if 's3://' in self.args['output_path']:
            AWS_param = AWS_Parameters(profile=self.args['profile'], region=self.args['region'], endpoint_url=endpoint)      
            kvstore_out = create_kvstore(fpath=str(out_path), store='s3', AWS_param=AWS_param)                                                
        
        
        input_arr = open_tensor(in_path, kvstore=kvstore_in, bytes_limit= 100_000_000, driver='zarr')
        chunk_shape=[1, 1, 64, 64, 64]
        if len(input_arr.shape)==3:
            chunk_shape=[64, 64, 64]

        try:
            output_arr = create_tensor(fpath=out_path, arr_shape=input_arr.shape, dtype='uint64', chunk_shape=chunk_shape, driver='zarr3', codecs={"name": "blosc", "configuration": {"cname": "lz4", "clevel": 4}}, sharded=True, kvstore=kvstore_out, shard_factor=16)
                                    
        except:
            output_arr = open_tensor(out_path, bytes_limit= 100_000_000, driver='zarr', kvstore=kvstore_out)  
            
                                                                                                                                               
        voxel_relabel_gunpowder(input_arr, output_arr, skel_path= self.args['skel_path'], iter_size=(64,64,64), batch_size=5, cutout=self.args['cutout'])          
                
                           


if __name__ == "__main__":
    mod = VoxelRelabelModule()
    mod.run()

__all__ = [
    "VoxelRelabelModule",
    "VoxelRelabelParameters"]
