import tensorstore as ts
from ac_segmentation.utils.tensorstore import open_tensor, AWS_Parameters, create_kvstore, create_tensor
from ac_segmentation.utils.preprocess import create_chunked_dims, create_overlap_chunks
import numpy as np
from scipy import ndimage
import gc
import multiprocessing as mp
from collections.abc import MutableMapping
from typing import Union
import warnings
import logging
import copy
import torch 
import os
from datetime import datetime
import tensorstore as ts
import itertools
import argschema
import json
import time
from ac_segmentation.methods.nodes import TensorStoreSource, Fuse, total_volume_shape


from ac_segmentation.gunpowder.array_spec import ArraySpec
from ac_segmentation.gunpowder.array import ArrayKey
from ac_segmentation.gunpowder.coordinate import Coordinate
from ac_segmentation.gunpowder.batch_request import BatchRequest
from ac_segmentation.gunpowder.roi import Roi
from ac_segmentation.gunpowder.build import build
from ac_segmentation.gunpowder.nodes.scan import Scan



logger = logging.getLogger(__name__)



def fuse_gunpowder(arrs, translations, output_path, flatten={'surface_maps':None, 'downsample':16, 'axis':'x', 'pad':None}, dtype='float32', iter_size = (64,64,64), batch_size=10, run_exclusive=-1, cutouts=None):

    torch.set_num_threads(20) 

    # Compute global bounding box
    mins, maxs = [], []
    for A, (x, y, z) in zip(arrs, translations):
        X, Y, Z = A.shape[-3:]
        mins.append([x, y, z])
        maxs.append([x+X, y+Y, z+Z])
    mins = np.min(mins, axis=0)
    maxs = np.max(maxs, axis=0)
    Xf, Yf, Zf = (maxs - mins).astype(int)
    
    # Detect dimensionality
    is_5d = arrs[0].ndim == 5
    if is_5d:
        t, c = arrs[0].shape[:2]

    # Get max x offset
    max_offset = 0
    if flatten['surface_maps']:
        for ind in range(len(flatten['surface_maps'])):
            smap = flatten['surface_maps'][ind].astype('int16')
            dsfactor = flatten['downsample']
            if dsfactor != 1:
                up = ndimage.zoom(smap, tuple([dsfactor]*2))
                flatten['surface_maps'][ind] = up + abs(int(up.min()))
            if flatten['surface_maps'][ind].max() > max_offset:
                max_offset = flatten['surface_maps'][ind].max()
                
        Xf += flatten['pad']
        Yf += flatten['pad']
        Zf += flatten['pad']                                                
        
        if flatten['axis'] == 'x':
            Xf += max_offset
        elif flatten['axis'] == 'y':
            Yf += max_offset
        elif flatten['axis'] == 'z':
            Zf += max_offset
    else:
        flatten['surface_maps'] = [None]*len(arrs)

    # Create output tensor — shape depends on dimensionality
    if is_5d:
        out_shape = (t, c, Xf, Yf, Zf)
        chunk_shape = [t, c, 64, 64, 64]
    else:
        out_shape = (Xf, Yf, Zf)
        chunk_shape = [64, 64, 64]
        
    try:
        out_arr = create_tensor(
            fpath=output_path,
            arr_shape=out_shape,
            chunk_shape=chunk_shape,
            res=[1,1,1,1,1] if is_5d else [1,1,1],
            dtype=dtype,
            codecs=None)
    except:
        out_arr = open_tensor(output_path)
        
    stime = datetime.now()
    
    # Define the chunk size and overlap
    if is_5d:
        iter_size_full = (1, 1) + tuple(iter_size)
        start_req = (0, 0, 0, 0, 0)
    else:
        iter_size_full = tuple(iter_size)
        start_req = (0, 0, 0)
    
    iter_size_full = np.minimum(np.array(iter_size_full), np.array(arrs[0].shape))
    chunk_size = batch_size * iter_size_full
    
    stime_og = datetime.now()
    
    # Iterate over each array
    for ind, (A, (x, y, z), smap) in enumerate(zip(arrs, translations, flatten['surface_maps'])):    
        if run_exclusive != -1 and ind != run_exclusive:
            pass
        
        else:     
            X, Y, Z = A.shape[-3:]
            x0_adj = x - mins[0]
            y0_adj = y - mins[1]
            z0_adj = z - mins[2]
    
            # Generate all block coordinates
            blocks = []
            start, end = create_chunked_dims(arr_shape=A.shape, chunk_size=chunk_size)
            
            for i in range(len(start)):
                start[i], end[i] = np.minimum(start[i], np.array(A.shape)), np.minimum(end[i], np.array(A.shape))
                dif = np.array(end[i]) - np.array(start[i])
                if np.any(dif[-3:] < iter_size_full[-3:]) == True:
                    x, y, z = np.array(end[i][-3:]) - np.array(iter_size_full[-3:])
                    start[i][-3:] = [x, y, z]      
                    
            if cutouts:
                start_new, end_new = [], []
                x1, x2, y1, y2, z1, z2 = cutouts[ind]
                print(cutouts[ind])
                for i, (s, e) in enumerate(zip(start, end)):
                    offset = np.array([x1, y1, z1])
                    s, e = np.array(s[-3:]) + offset, np.array(e[-3:]) + offset
                    if (s[0] <= x2 and s[1] <= y2 and s[2] <= z2):
                        e[-3:] = np.minimum(np.array([x2, y2, z2]) + (iter_size_full[-1]/2), np.array(e[-3:]))
                        if is_5d:
                            s = np.concatenate(([0, 0], s))
                            e = np.concatenate(([1, 1], e))                      
                        start_new.append(s)
                        end_new.append(e)                           
                start, end = start_new, end_new
                                                                                             
            print("Processing {0} blocks".format(len(start)))
    
            raw = ArrayKey('RAW')
            dtype = A.dtype.name
            
            source = TensorStoreSource(
                {raw: A},
                {raw: ArraySpec(interpolatable=True)}
            )
            
            iter_coord = Coordinate(iter_size_full)
            
            scan_request = BatchRequest()
            scan_request[raw] = Roi(start_req, iter_coord)
            scan = Scan(scan_request) 
            
            fuse = Fuse(raw, out_arr, x0_adj, y0_adj, z0_adj, {'surface_map': smap, 'downsample': flatten['downsample'], 'axis': flatten['axis']})
            
            pipeline = (source + fuse + scan)
                
            with build(pipeline):
                stime = datetime.now()
             
                for i in range(len(start)):
                    arr = np.array(end[i]) - np.array(start[i])
                    total_roi = Roi(np.array(start[i]), arr)
            
                    request = BatchRequest()
                    request[raw] = total_roi
                    batch = pipeline.request_batch(request)
                  
                    final_write = []
                    write_objects = fuse.get_write_objects()
                    if len(write_objects) > 0:
                        indices = [item[0] for item in write_objects]
                        x1,x2,y1,y2,z1,z2 = [max(slot) for slot in zip(*indices)]
                        mx1,mx2,my1,my2,mz1,mz2 = [min(slot) for slot in zip(*indices)]
                        
                        temp_shape = [x2-mx1, y2-my1, z2-mz1]
                        if is_5d:
                            temp_shape = [1, 1] + temp_shape
                        
                        success = False
                        max_retries = 10
                        
                        for attempt in range(max_retries):
                            try:
                                temp_arr = np.zeros(temp_shape, dtype=dtype)
                                for write in write_objects:
                                    ox1,ox2,oy1,oy2,oz1,oz2 = write[0]
                                    if is_5d:
                                        temp_arr[:, :, ox1-mx1:ox2-mx1, oy1-my1:oy2-my1, oz1-mz1:oz2-mz1] = write[1][None, None, :]
                                    else:
                                        temp_arr[ox1-mx1:ox2-mx1, oy1-my1:oy2-my1, oz1-mz1:oz2-mz1] = write[1]
                                    
                                if is_5d:
                                    existing_block = out_arr[:, :, mx1:x2, my1:y2, mz1:z2].read().result()
                                    if np.any(existing_block):
                                        temp_arr = np.maximum(existing_block, temp_arr)
                                    out_arr[:, :, mx1:x2, my1:y2, mz1:z2].write(temp_arr[:, :, :]).result()
                                else:
                                    existing_block = out_arr[mx1:x2, my1:y2, mz1:z2].read().result()
                                    if np.any(existing_block):
                                        temp_arr = np.maximum(existing_block, temp_arr)
                                    out_arr[mx1:x2, my1:y2, mz1:z2].write(temp_arr).result()
                                success = True
                                break
                            except Exception as e:
                                print(f"Attempt {attempt+1} failed: {e}")
                                if attempt < max_retries - 1:
                                    time.sleep(5)
                        
                    fuse.clear_write_objects() 
                    etime = datetime.now()
                    print(i, etime - stime)
                    
    etime = datetime.now()
    print("Total run time: ", etime - stime_og)
                


class FusionZarrParameters(argschema.ArgSchema):
    in_paths = argschema.fields.String(required=True)
    output_path = argschema.fields.String(required=True)
    translations = argschema.fields.String(required=True)
    run_exclusive = argschema.fields.Int(default=-1)   
    mip  =  argschema.fields.Int(default=4)     
    
    AWS_key = argschema.fields.String(required=False, default=None, allow_none=True)
    AWS_sec_key = argschema.fields.String(required=False, default=None, allow_none=True)
    region = argschema.fields.String(required=False, default='us-west-2')
    endpoint = argschema.fields.String(required=False, default=None, allow_none=True)
    profile = argschema.fields.String(required=False, default=None, allow_none=True)

    
class FusionZarrModule(argschema.ArgSchemaParser):
    default_schema = FusionZarrParameters
            

    def run(self):
        fpaths = self.args['in_paths']
        
        fpaths = open(self.args['in_paths'], "r").read().splitlines()
        
        with open(self.args['translations'], 'r') as file:
            translations = json.load(file)
            
                    
        #open input tensors
        arrays = []
        kvstore_in, kvstore_out = None, None                       
        if not self.args['endpoint']:
            endpoint=None
        for fpath in fpaths:
            #Open input tensor  
            in_path = os.path.join(fpath, str(self.args['mip']))                
                
            if 's3://' in in_path:
                if self.args['profile']:
                    AWS_param = AWS_Parameters(profile=self.args['profile'], region=self.args['region'], endpoint_url=self.args['endpoint'])      
                    kvstore_in = create_kvstore(fpath=str(in_path), store='s3', AWS_param=AWS_param)              
                                        
                if self.args['AWS_key']:
                    AWS_param = AWS_Parameters(region=self.args['region'], endpoint_url=self.args['endpoint'])
                    AWS_param.add_credentials(access_key_id=self.args['AWS_key'], secret_access_key=self.args['AWS_sec_key'])
                    kvstore_in = create_kvstore(fpath=str(in_path), store='s3', AWS_param=AWS_param)                        
                                               
            arrays.append(open_tensor(in_path, kvstore=kvstore_in)) 
                                   
                            
        fuse_gunpowder(arrs=arrays, translations=translations, output_path=os.path.join(self.args['output_path'], str(self.args['mip'])), batch_size=5, iter_size = (64,64,64), run_exclusive=self.args['run_exclusive']) 
    


if __name__ == "__main__":
    mod = FusionZarrModule()
    mod.run()

__all__ = [
    "FusionZarrModule",
    "FusionZarrParameters"]