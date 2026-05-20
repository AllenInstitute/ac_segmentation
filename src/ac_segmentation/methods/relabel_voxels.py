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

    
    raw = ArrayKey('RAW')
    source = TensorStoreSource(
        {
            raw: input_arr
        },
        {
            raw: ArraySpec(interpolatable=True)
        })

    
    start_req = (0,0,0)
    if len(input_arr.shape) == 5:
        iter_size = (1,1,) + iter_size
        start_req = (0,0,0,0,0)
    chunk_size = batch_size*(np.array(iter_size))
    iter_coord = Coordinate(iter_size)

    # Define the scan request with overlap
    scan_request = BatchRequest()
    scan_request[raw] = Roi(start_req, iter_coord)
    scan = Scan(scan_request, num_workers=0)
    
    #create chunks
    start, end = create_chunked_dims(arr_shape=input_arr.shape, chunk_size=chunk_size)
    
    if cutout:
        start_new, end_new = [], []
        x1, x2, y1, y2, z1, z2 = cutout
        for i, (s, e) in enumerate(zip(start, end)):
            offset = np.array([x1,y1,z1])
            s, e = np.array(s[-3:])+offset, np.array(e[-3:])+offset
            if (s[0] <= x2 and s[1] <= y2 and s[2] <= z2):
                e[-3:] = np.minimum(np.array([x2,y2,z2])+(iter_size[-1]/2), np.array(e[-3:]))
                if len(input_arr.shape) == 5:
                    s = np.concatenate(([0, 0], s))
                    e = np.concatenate(([1, 1], e))                      
                start_new.append(s)
                end_new.append(e)
                    
        start, end = start_new, end_new

         
    for i in range(len(start)):
        start[i], end[i] = np.minimum(start[i], np.array(input_arr.shape)), np.minimum(end[i], np.array(input_arr.shape))
        dif = np.array(end[i]) - np.array(start[i])
        if np.any(dif[-3:] < iter_size[-1]) == True:
            x,y,z = np.array(end[i][-3:])-np.array(iter_size[-3:])
            start[i][-3:] = [x,y,z]
            
        
    if len(start) == 0:
        print('Batch_size needs to be lowered to accommodate the cutout size.')
        return
    
    

    stime = datetime.now()   
    all_skels, shards = query_skeletons_by_bb([x1,y1,z1,x2,y2,z2], skel_path, n_workers=10)

    for ind,i in enumerate(range(len(start))): 
    
        #query skeletons
        sx1,sy1,sz1 = start[i][-3:]
        sx2,sy2,sz2 = end[i][-3:]
        
        skels = filter_skeletons(all_skels,[sx1,sx2,sy1,sy2,sz1,sz2])
                
        if len(skels)>0:
            print(start[i], end[i], "# Skels: ", len(skels))
        
            relabel = VoxelRelabel(raw,raw,input_arr,output_arr, skels)
                
            # Build the pipeline with Scan
            pipeline = (
                    source +
                    relabel +
                    scan)
    
            with build(pipeline):
                arr = np.array(end[i])-np.array(start[i])
                total_roi = Roi(start[i], arr)
                
        
                # Create a request for the entire volume
                request = BatchRequest()
                request[raw] = total_roi
                
                # Request the batch
                batch = pipeline.request_batch(request)
                    
                # Retrieve the list of write objects and write them
                write_objects = relabel.get_write_objects()
                if len(write_objects) > 0:
                    indices = [item[0] for item in write_objects]
                    x1,x2,y1,y2,z1,z2 = [max(slot) for slot in zip(*indices)]
                    mx1,mx2,my1,my2,mz1,mz2 = [min(slot) for slot in zip(*indices)]
                        
                    if len(output_arr.shape) == 5:
                        temp_shape = [1,1] + [x2-mx1,y2-my1,z2-mz1]
    
                    success = False
                    max_retries = 10
                    for attempt in range(max_retries):
                        try:
                            temp_arr = output_arr[:,:,mx1:x2,my1:y2,mz1:z2].read().result()
                            for write in write_objects:
                                ox1,ox2,oy1,oy2,oz1,oz2 = write[0]
                                arr2 = temp_arr[0,0,ox1-mx1:ox2-mx1, oy1-my1:oy2-my1, oz1-mz1:oz2-mz1]
                                write_data = np.maximum(arr2,write[1])
                                temp_arr[:,:,ox1-mx1:ox2-mx1, oy1-my1:oy2-my1, oz1-mz1:oz2-mz1] = write_data[None, None, :]
                            
                            if np.any(temp_arr):
                                if len(input_arr.shape) == 5:
                                    output_arr[:, :, mx1:x2, my1:y2, mz1:z2].write(temp_arr[:, :, :, :, :]).result()
                                else:
                                    output_arr[mx1:x2, my1:y2, mz1:z2].write(temp_arr[:, :, :]).result()
                            success = True
                            break  # Exit loop if successful
                    
                        except Exception as e:
                            print(f"Attempt {attempt+1} failed: {e}")
                            if attempt < max_retries - 1:
                                time.sleep(5)  # Wait before next retry
                    
    
                    relabel.clear_write_objects()
                    etime = datetime.now()
                    print(i, etime - stime)
        else:
            print(start[i], end[i], 'no skeletons')


                                                                                
class VoxelRelabelParameters(argschema.ArgSchema):
    input_path = argschema.fields.String(required=True)
    skel_path = argschema.fields.String(required=True)
    output_path = argschema.fields.String(required=True)
    cutout = argschema.fields.Raw(required=False, allow_none=True, missing=None)

    region = argschema.fields.String(required=False, default='us-west-2')
    endpoint = argschema.fields.String(required=False, default=None, allow_none=True)
    profile = argschema.fields.String(required=False, default=None, allow_none=True)
    

class VoxelRelabelModule(argschema.ArgSchemaParser):
    default_schema = VoxelRelabelParameters
       

    def run(self):
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

        try:
            output_arr = create_tensor(fpath=out_path, arr_shape=input_arr.shape, dtype='uint64', chunk_shape=[1, 1, 64, 64, 64], driver='zarr3', codecs={"name": "blosc", "configuration": {"cname": "lz4", "clevel": 4}}, sharded=True, kvstore=kvstore_out, shard_factor=16)
                                    
        except:
            output_arr = open_tensor(out_path, bytes_limit= 100_000_000, driver='zarr', kvstore=kvstore_out)  
            
                                                                                                                                               
        voxel_relabel_gunpowder(input_arr, output_arr, skel_path= self.args['skel_path'], iter_size=(64,64,64), batch_size=5, cutout=self.args['cutout'])          
                
                           


if __name__ == "__main__":
    mod = VoxelRelabelModule()
    mod.run()

__all__ = [
    "VoxelRelabelModule",
    "VoxelRelabelParameters"]