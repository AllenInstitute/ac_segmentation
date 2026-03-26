from ac_segmentation.gunpowder.nodes import TensorStoreSource, ContrastAdjustWrite, perimeter_weighted_blend
from ac_segmentation.utils.preprocess import create_chunked_dims, create_overlap_chunks
from ac_segmentation.utils.tensorstore import open_tensor, AWS_Parameters, create_kvstore, create_tensor
from ac_segmentation.gunpowder.bump_mask import *

import gunpowder as gp
import os
import numpy as np
from datetime import datetime
import json
import time
import argschema


def adjust_contrast_gunpowder(input_arr, output_arr, iter_size=(64,64,64), batch_size=3, cutout=None, mask_file=None, preprocess={'method':'percentile','values':[96,97]}, dsfactor=1, add_margin=0, depth=.7):

    mask=None
    if mask_file:
        mask = open_tensor(fpath=mask_file).read().result()
        print('mask', mask.shape)
    
    raw = gp.ArrayKey('RAW')
    source = TensorStoreSource(
        {
            raw: input_arr
        },
        {
            raw: gp.ArraySpec(interpolatable=True)
        }, add_margin=add_margin)

    
    start_req = (0,0,0)
    if len(input_arr.shape) == 5:
        iter_size = (1,1,) + iter_size
        start_req = (0,0,0,0,0)
    chunk_size = batch_size*(np.array(iter_size))
    iter_coord = gp.Coordinate(iter_size)

    # Define the scan request with overlap
    scan_request = gp.BatchRequest()
    scan_request[raw] = gp.Roi(start_req, iter_coord)
    scan = gp.Scan(scan_request, num_workers=0)
    
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
        print(start[i],end[i])
         
    for i in range(len(start)):
        start[i], end[i] = np.minimum(start[i], np.array(input_arr.shape)), np.minimum(end[i], np.array(input_arr.shape))
        dif = np.array(end[i]) - np.array(start[i])
        if np.any(dif[-3:] < iter_size[-1]) == True:
            x,y,z = np.array(end[i][-3:])-np.array(iter_size[-3:])
            start[i][-3:] = [x,y,z]
            
        
    if len(start) == 0:
        print('Batch_size needs to be lowered to accommodate the cutout size.')
        return
    
    method, values = preprocess['method'], preprocess['values']
    contrast = ContrastAdjustWrite(raw,raw,input_arr,output_arr,int_range=values,version=method, mask=mask, dsfactor=dsfactor, add_margin=add_margin)
        
    # Build the pipeline with Scan
    pipeline = (
            source +
            contrast +
            scan)

    stime = datetime.now()

    print(len(start))

    with gp.build(pipeline):
        for ind,i in enumerate(range(len(start))):
            print(start[i], end[i])
            arr = np.array(end[i])-np.array(start[i])
            total_roi = gp.Roi(start[i], arr)
    
            # Create a request for the entire volume
            request = gp.BatchRequest()
            request[raw] = total_roi
            
            # Request the batch
            batch = pipeline.request_batch(request)
                
            # Retrieve the list of write objects and write them
            write_objects = contrast.get_write_objects()
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
                            write_data = perimeter_weighted_blend(arr2, write[1], depth=depth)#.astype('uint8')
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
                

                contrast.clear_write_objects()
                etime = datetime.now()
                print(i, etime - stime)


                                                                                
class ContrastParameters(argschema.ArgSchema):
    input_path = argschema.fields.String(required=True)
    output_path = argschema.fields.String(required=True)
    cutout = argschema.fields.Raw(required=False, allow_none=True, missing=None)
    dsfactor = argschema.fields.Float(required=False, default=16, allow_none=True)
    mask_path = argschema.fields.String(required=False, allow_none=True, default='None')
    
    AWS_key = argschema.fields.String(required=False, default=None, allow_none=True)
    AWS_sec_key = argschema.fields.String(required=False, default=None, allow_none=True)
    region = argschema.fields.String(required=False, default='us-west-2')
    endpoint = argschema.fields.String(required=False, default=None, allow_none=True)
    profile = argschema.fields.String(required=False, default=None, allow_none=True)
    

class ContrastModule(argschema.ArgSchemaParser):
    default_schema = ContrastParameters
       

    def run(self):
        for key, value in self.args.items():
            if value == 'None':
                self.args[key] = None
                
        # --- Convert bound_box from string to list if present ---
        if self.args['cutout'] and type(self.args['cutout'])==str:
            self.args['cutout'] = [int(x.strip("'")) for x in self.args["cutout"].split(',')]
                                                     
             
        kvstore_in, kvstore_out = None, None                                   
        for mip in range(0,5):
            in_path = os.path.join(str(self.args['input_path']), str(mip))
            out_path = os.path.join(str(self.args['output_path']), str(mip))
            
            
            if not self.args['endpoint']:
                endpoint=None
                
            if 's3://' in self.args['input_path']:
                if self.args['profile']:
                    AWS_param = AWS_Parameters(profile=self.args['profile'], region=self.args['region'], endpoint_url=self.args['endpoint'])      
                    kvstore_in = create_kvstore(fpath=str(in_path), store='s3', AWS_param=AWS_param)              
                                        
                if self.args['AWS_key']:
                    AWS_param = AWS_Parameters(region=self.args['region'], endpoint_url=self.args['endpoint'])
                    AWS_param.add_credentials(access_key_id=self.args['AWS_key'], secret_access_key=self.args['AWS_sec_key'])
                    kvstore_in = create_kvstore(fpath=str(in_path), store='s3', AWS_param=AWS_param)
            
            if 's3://' in self.args['output_path']:
                if self.args['profile']:
                    AWS_param = AWS_Parameters(profile=self.args['profile'], region=self.args['region'], endpoint_url=self.args['endpoint'])      
                    kvstore_out = create_kvstore(fpath=str(out_path), store='s3', AWS_param=AWS_param)              
                                        
                if self.args['AWS_key']:                           
                    AWS_param = AWS_Parameters(region=self.args['region'], endpoint_url=self.args['endpoint'])
                    AWS_param.add_credentials(access_key_id=self.args['AWS_key'], secret_access_key=self.args['AWS_sec_key'])
                    kvstore_out = create_kvstore(fpath=str(out_path), store='s3', AWS_param=AWS_param)                                                
            
            size= max(4,int((90/(2**mip))))
            iter_size = tuple(np.array([size]*3))
            add_margin = int(np.ceil(size/1.75))
            batch_size = int((500/size))
            shard_factor = int(16/(mip+1))
            ds_factor = self.args['dsfactor']/(2**mip)
                           
            
            input_arr = open_tensor(in_path, kvstore=kvstore_in, bytes_limit= 100_000_000, driver='zarr')

            try:
                output_arr = create_tensor(fpath=out_path, arr_shape=input_arr.shape, dtype='uint8', chunk_shape=[1, 1, 64, 64, 64], driver='zarr3', codecs={"name": "blosc", "configuration": {"cname": "lz4", "clevel": 4}}, sharded=True, kvstore=kvstore_out, shard_factor=shard_factor)
                                        
            except:
                output_arr = open_tensor(out_path, bytes_limit= 100_000_000, driver='zarr', kvstore=kvstore_out)  
                
                                                                                                                                                   
            adjust_contrast_gunpowder(input_arr, output_arr, iter_size=iter_size, batch_size=batch_size, cutout=self.args['cutout'], preprocess={'method':'percentile','values':[5,99.5]}, mask_file=self.args['mask_path'], dsfactor=ds_factor, add_margin=add_margin, depth=.9)          
                
                           


if __name__ == "__main__":
    mod = ContrastModule()
    mod.run()

__all__ = [
    "ContrastModule",
    "ContrastParameters"]