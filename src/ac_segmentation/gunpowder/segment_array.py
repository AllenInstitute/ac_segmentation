import ac_segmentation.neurotorch.nets.RSUNet
from ac_segmentation.gunpowder.nodes import TensorStoreSource, ApplyModel, ContrastAdjust
from ac_segmentation.utils.tensorstore import open_tensor, AWS_Parameters, create_kvstore, create_tensor
from ac_segmentation.utils.preprocess import create_chunked_dims, create_overlap_chunks

import tensorstore as ts
import gunpowder as gp
import torch
import os
import numpy as np
from datetime import datetime
import argschema
import pathlib
import docker
import torch
import ast


RSUNet = ac_segmentation.neurotorch.nets.RSUNet.RSUNet

    
def segment_gunpowder(input_arr, output_arr, checkpoint, iter_size=(64,64,64), batch_size=5, cutout=None, gpu_device=None, cpus=20, preprocess={'method':'percentile','values':[96,97]}, mask_file=None, dsfactor=1, add_margin=0):
    mask=None
    if mask_file:
        mask = open_tensor(fpath=mask_file).read().result()

    raw = gp.ArrayKey('RAW')
    source = TensorStoreSource(
        {
            raw: input_arr
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
        }, add_margin=add_margin)

    
    if int(cpus) > int(os.cpu_count()):
        cpus = os.cpu_count()
    torch.set_num_threads(cpus) 
    
    # Set-up model
    device = torch.device("cuda:{}" .format(gpu_device) if gpu_device is not None else "cpu")
    model = RSUNet()
    
    model = model.to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
        

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

    if add_margin:
        iter_size = tuple(np.array(iter_size) + add_margin)
    
    # Create the ApplyModel instance
    apply_model = ApplyModel(model, raw, output_arr, device, mask=mask, dsfactor=dsfactor, add_margin=add_margin)
    
    # Build the pipeline with Scan
    method, values = preprocess['method'], preprocess['values']
    pipeline = (
        source +
        ContrastAdjust(raw,raw,values,version=method) +
        apply_model +
        scan
    )
    
    stime = datetime.now()

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

    #run pipeline    
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
            final_write = []
            write_objects = apply_model.get_write_objects()
            if len(write_objects) > 0:
                indices = [item[0] for item in write_objects]
                x1,x2,y1,y2,z1,z2 = [max(slot) for slot in zip(*indices)]
                mx1,mx2,my1,my2,mz1,mz2 = [min(slot) for slot in zip(*indices)]
                
                temp_shape = [x2-mx1,y2-my1,z2-mz1]
                if len(output_arr.shape) == 5:
                    temp_shape = [1,1] + temp_shape
                
                success = False
                max_retries = 10
                
                for attempt in range(max_retries):
                    try:
                        temp_arr = np.zeros(temp_shape, dtype='uint8')
                        for write in write_objects:
                            ox1,ox2,oy1,oy2,oz1,oz2 = write[0]
                            arr2 = temp_arr[0,0,ox1-mx1:ox2-mx1, oy1-my1:oy2-my1, oz1-mz1:oz2-mz1]
                            write[1] = (write[1]*255).astype('uint8')
                            write_data = np.maximum(arr2, write[1])
                            temp_arr[:,:,ox1-mx1:ox2-mx1, oy1-my1:oy2-my1, oz1-mz1:oz2-mz1] = write_data[None, None, :]
                            
                        if len(input_arr.shape) == 5:
                            final_write.append(output_arr[:, :, mx1:x2, my1:y2, mz1:z2].write(temp_arr[:, :, :, :, :]))
                        else:
                            final_write.append(output_arr[mx1:x2, my1:y2, mz1:z2].write(temp_arr[:, :, :]))
                        success = True
                        break  # Exit loop if successful
                    except Exception as e:
                        print(f"Attempt {attempt+1} failed: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(5)  # Wait before next retry
                
                apply_model.clear_write_objects()
                etime = datetime.now()
                print(i, etime - stime)
    
    for write in final_write:
        write.result()
                
    etime = datetime.now()
    print(etime-stime)
    
    #output seg card
    compute = {'GPUs': gpu_device} if gpu_device else {'CPUs': cpus}
    seg_card = {'date':datetime.today().strftime('%Y-%m-%d'),
                'paths':{'inpath':input_arr.kvstore.path,'outpath':output_arr.kvstore.path}, 
                'preprocessing':{'method':method,'values':values}, 
                'compute':compute,
                'time_lapse':etime-stime}

    with open(os.path.join(output_arr.kvstore.path, "seg_card.txt"), 'w') as f:
        f.write(str(seg_card))
    
    return seg_card
      
class SegmentZarrParameters(argschema.ArgSchema):
    input_zarr = argschema.fields.String(required=True)
    weights_file = argschema.fields.InputFile(required=True)
    probability_output = argschema.fields.String(required=True)         
    filter_max_intensity = argschema.fields.Int(required=False, default=30000, allow_non=True)
    rescale_perc = argschema.fields.String(allow_none=True, default='[10,99]')
    cutout = argschema.fields.Raw(required=False, allow_none=True, missing=None)
    dsfactor = argschema.fields.Int(required=False, default=1, allow_none=True)
    mask_path = argschema.fields.String(required=False, allow_none=True, missing=None)
    iter_size = argschema.fields.Int(required=False, default=50, allow_none=True)
    
    AWS_key = argschema.fields.String(required=False, default=None, allow_none=True)
    AWS_sec_key = argschema.fields.String(required=False, default=None, allow_none=True)
    region = argschema.fields.String(required=False, default='us-west-2')
    endpoint = argschema.fields.String(required=False, default=None, allow_none=True)
    profile = argschema.fields.String(required=False, default=None, allow_none=True)
    
class SegmentZarrModule(argschema.ArgSchemaParser):
    default_schema = SegmentZarrParameters
            

    def run(self):

        # --- Convert all "None" strings to actual None ---
        for key, value in self.args.items():
            if value == "None":
                self.args[key] = None

        # --- Convert bound_box from string to list if present ---
        if self.args['cutout'] and type(self.args['cutout'])==str:
            self.args['cutout'] = [int(x.strip("'")) for x in self.args["cutout"].split(',')]
           
        
        kvstore_in, kvstore_out = None, None
        if 's3' in self.args['input_zarr']:
            if self.args['profile']:
                AWS_param = AWS_Parameters(profile=self.args['profile'], region=self.args['region'], endpoint_url=self.args['endpoint'])      
                kvstore_in = create_kvstore(fpath=str(self.args['input_zarr']), store='s3', AWS_param=AWS_param)              
                                        
            if self.args['AWS_key']:
                AWS_param = AWS_Parameters(region=self.args['region'], endpoint_url=self.args['endpoint'])
                AWS_param.add_credentials(access_key_id=self.args['AWS_key'], secret_access_key=self.args['AWS_sec_key'])
                kvstore_in = create_kvstore(fpath=str(self.args['input_zarr']), store='s3', AWS_param=AWS_param)
                
            
        if 's3' in self.args['probability_output']:
            if self.args['profile']:
                AWS_param = AWS_Parameters(profile=self.args['profile'], region=self.args['region'], endpoint_url=self.args['endpoint'])      
                kvstore_out = create_kvstore(fpath=self.args['probability_output'], store='s3', AWS_param=AWS_param)              
                                        
            if self.args['AWS_key']:                           
                AWS_param = AWS_Parameters(region=self.args['region'], endpoint_url=self.args['endpoint'])
                AWS_param.add_credentials(access_key_id=self.args['AWS_key'], secret_access_key=self.args['AWS_sec_key'])
                kvstore_out = create_kvstore(fpath=self.args['probability_output'], store='s3', AWS_param=AWS_param)
        else:
            os.makedirs(self.args['probability_output'],exist_ok=True)               
        

        # --- Open input tensor ---
        input_arr = open_tensor(
            self.args['input_zarr'].rstrip('/'),
            bytes_limit=100_000_000,
            driver='zarr', kvstore=kvstore_in)
            
        chunk_shape = [64, 64, 64]
        if len(input_arr.shape) == 5:
            chunk_shape = [1, 1, 64, 64, 64]

        # --- Open or create output tensor ---
        try:
            output_arr = create_tensor(
                fpath=self.args['probability_output'],
                arr_shape=input_arr.shape,
                dtype='uint8',
                chunk_shape=chunk_shape,
                driver='zarr3',
                codecs={"name": "blosc", "configuration": {"cname": "lz4", "clevel": 4}},
                sharded=True,
                shard_factor=16,
                kvstore=kvstore_out
            )
        except:
            output_arr = open_tensor(
                self.args['probability_output'],
                bytes_limit=100_000_000,
                driver='zarr',
                kvstore=kvstore_out
            )

        # --- Preprocess ---
        if self.args.get('rescale_perc'):
            rescale = ast.literal_eval(self.args["rescale_perc"])
            preprocess = {'method': 'percentile', 'values': rescale}
        else:
            preprocess = {'method': 'range', 'values': [0, 60000]}

        # --- Run segmentation ---
        segment_gunpowder(
            input_arr,
            output_arr,
            self.args["weights_file"],
            iter_size=tuple([self.args["iter_size"]]*3),
            batch_size=5,
            cutout=self.args.get("cutout"),
            gpu_device=None,
            cpus=40,
            preprocess=preprocess,
            mask_file=self.args.get('mask_path'),
            dsfactor=self.args.get('dsfactor'),
            add_margin=16  
        )



if __name__ == "__main__":
    mod = SegmentZarrModule()
    mod.run()

__all__ = [
    "SegmentZarrModule",
    "SegmentZarrParameters"]
    
    