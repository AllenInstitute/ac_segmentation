import pandas as pd
import numpy as np
import zarr
import docker
import tensorstore as ts
import os
import json
import boto3

def create_tensor(fpath, arr_shape, driver='zarr', store='file', dtype='float32', fill_value=-np.inf, 
                       chunk_shape=[64, 64, 64], res=[1,1,1], scale=0, arr=None, AWS_Key=None, AWS_Secret_Key=None):
    """Create a tensorstore object, with optional setting of array
       driver: Type of file, including zarr, n5, precomputed
       store: Type of source, including file, in-memory, s3
       AWS Key, AWS_Secret_Key: Only applicable to s3 store
    """
    if 'int' in str(dtype):
        fill_value=0
    if isinstance(arr, np.ndarray):
        arr = arr.astype(dtype)
    kvstore = {"driver": store,"path": fpath}
    if store == 's3':
        if not AWS_Key or not  AWS_Secret_Key:
            raise TypeError("AWS_Key and AWS_Secret_Key required for the S3 store")
        os.environ['AWS_ACCESS_KEY_ID'], os.environ['AWS_SECRET_ACCESS_KEY']=AWS_Key, AWS_Secret_Key
        bucket = fpath.split("/")[0]
        path = fpath.replace(bucket+"/", '')
        kvstore = {"driver": "s3","bucket": bucket ,"path": path}

    if driver in ['zarr','n5']:
        fill_value=None if driver=='n5' else fill_value
        out_arr = ts.open({
         'driver': driver,
         'kvstore': kvstore,
         },
         dtype=dtype,
         fill_value=fill_value,
         chunk_layout=ts.ChunkLayout(chunk_shape=chunk_shape),
         create=True,
         shape=list(arr_shape)).result()

    if driver == 'neuroglancer_precomputed':
        arr_shape=list(arr_shape)+[1] if len(arr_shape)==3 else arr_shape
        out_arr = ts.open(
                    {
                        "driver": "neuroglancer_precomputed",
                        "kvstore": kvstore,
                        "scale_metadata": {
                            "resolution": res,
                            "chunk_size": list(chunk_shape),
                            "encoding": "raw",
                            "key": "s" + str(scale)
                        }
                    },
                    create=True,
                    dtype=dtype,
                    domain=ts.IndexDomain(
                        shape=list(list(arr_shape)),
                    )).result()

    if isinstance(arr, np.ndarray):
        out_arr.write(arr).result()
        
    os.environ['AWS_ACCESS_KEY_ID'], os.environ['AWS_SECRET_ACCESS_KEY'] = '', ''
    return out_arr

def open_tensor(fpath, driver='zarr', store='file', AWS_Key=None, AWS_Secret_Key=None, bytes_limit= 100_000_000):
    """Open a tensorstore object.
       driver: Type of file, including zarr, n5, precomputed
       store: Type of source, including file, s3
       AWS Key, AWS_Secret_Key: Only applicable to s3 store
    """

    kvstore = {"driver": store,"path": fpath}
    if store == 's3':
        if not AWS_Key or not  AWS_Secret_Key:
            raise TypeError("AWS_Key and AWS_Secret_Key required for the S3 store")
        os.environ['AWS_ACCESS_KEY_ID'], os.environ['AWS_SECRET_ACCESS_KEY']=AWS_Key, AWS_Secret_Key
        bucket = fpath.split("/")[0]
        path = fpath.replace(bucket+"/", '')
        kvstore = {"driver": "s3","bucket": bucket ,"path": path}
    #Load tensorstore array
    dataset_future = ts.open({
         'driver':
             driver,
         'kvstore': kvstore,
     # Use 100MB in-memory cache.
         'context': {
             'cache_pool': {
                 'total_bytes_limit': bytes_limit
             }
         },
         'recheck_cached_data':
         'open',
     })

    os.environ['AWS_ACCESS_KEY_ID'], os.environ['AWS_SECRET_ACCESS_KEY'] = '', ''
    return dataset_future.result()

create_EmptyTensor = create_tensor  
open_ZarrTensor = open_tensor
    
    
def zarr_to_n5(zarr_path, out_path, chunks=(64,64,64), cutout=None):
    #open zarr
    arr = open_ZarrTensor(zarr_path)
    if cutout != None:
        x1,x2,y1,y2,z1,z2 = cutout
        arr = arr[0,0,x1:x2,y1:y2,z1:z2].transpose().read().result()
    else:
        arr = arr[0,0,:,:,:].transpose().read().result()

    #create n5
    store = zarr.N5Store(os.path.join(out_path, '.n5'))
    root = zarr.group(store=store)
    z = root.zeros('group/' + zarr_path[-2], shape=arr.shape, chunks=chunks, dtype=arr.dtype, compressor=None)
    z[:] = arr
    
def zarr_to_precomputed(zarr_path, out_path, store='file', chunks=(64,64,64), cutout=None, scales=6, AWS_Key=None, AWS_Secret_Key=None):
    #iterate over all scale levels
    for scale in range(0,scales):
        #open zarr
        arr = open_tensor(zarr_path+str(scale))
        if cutout!=None:
            x1,x2,y1,y2,z1,z2 = cutout
            arr = arr[0,0,x1:x2,y1:y2,z1:z2].read().result()
            cutout = list((np.array(cutout)/2).astype(int))
        else:
            arr = arr[0,0,:,:,:].read().result()
        arr = np.expand_dims(arr, axis=3)
    
        #get resolution
        r_path = os.path.join(os.path.dirname(zarr_path), ".zattrs")
        res = json.loads(open(r_path, "r").read())['multiscales'][0]['datasets'][int(scale)]['coordinateTransformations'][0]['scale'][2:]
        
        #create precomputed tensor
        pre_comp = create_tensor(out_path, arr_shape=arr.shape, dtype=arr.dtype, store=store, driver='neuroglancer_precomputed', 
                                      AWS_Key=AWS_Key, AWS_Secret_Key=AWS_Secret_Key, scale=scale)
        pre_comp.write(arr).result()
        

def zarr_to_CATMAID_project(zarr_path, out_path, container_id, AWS_Key, AWS_Secret_Key, project='NewProject', 
                            stack='NewStack', store='s3', chunks=(64,64,64), translation=(0,0,0), cutout=None, tile_dim=[128,128], ortho=False):

    #convert zarr to precomputed
    zarr_to_precomputed(zarr_path=zarr_path, AWS_Key=AWS_Key, AWS_Secret_Key=AWS_Secret_Key, 
                           out_path=out_path, store=store, chunks=(64,64,64), cutout=cutout)
    
    #extract url 
    split = out_path.split("/")
    bucket = split[0]
    key = out_path.replace(bucket+"/", '')
    url = os.path.join("https://", bucket+".s3-us-west-2.amazonaws.com", key, "%SCALE_DATASET%/")
    
    #get resolution and shape
    shape = open_tensor(zarr_path + "0/")[0,0,:,:,:].shape
    if cutout!=None:
        x1,x2,y1,y2,z1,z2 = cutout
        x,y,z = x2-x1,y2-y1,z2-z1
        shape = [x,y,z]
    r_path = os.path.join(os.path.dirname(os.path.dirname(zarr_path + "0/")), ".zattrs")
    res = json.loads(open(r_path, "r").read())['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][2:]
    
    #create project data json
    if ortho==True:
      stack_file = [{
        "project": {
          "title": project,
          "stacks": [{
            "title": stack+"xy",
            "dimension": str(tuple(shape)),
            "mirrors": [{
              "fileextension": "raw",
              "position": 0,
              "tile_source_type": 14,
              "tile_height":tile_dim[0],
              "tile_width":tile_dim[1],
              "title": stack+"_tiles",
              "url": url + "0_1_2"
            }],
            "resolution": str(tuple(res)),
            "translation": str(translation),
            "downsample_factors": ["(1,1,1)", "(2,2,2)", "(4,4,4)"],
            "orientation" : 0,
            "stackgroups": [{"title": project+"_StackGroup", "relation": "channel"}]
          },
          {
            "title": stack+"xz",
            "dimension": str(tuple([shape[0],shape[2],shape[1]])),
            "mirrors": [{
              "fileextension": "raw",
              "position": 0,
              "tile_source_type": 14,
              "tile_height":tile_dim[0],
              "tile_width":tile_dim[1],
              "title": stack+"_tiles",
              "url": url + "0_2_1"
            }],
            "resolution": str(tuple([res[0],res[2],res[1]])),
            "translation": str(tuple([translation[0],translation[2],translation[1]])),
            "downsample_factors": ["(1,1,1)", "(2,2,2)", "(4,4,4)"],
            "orientation" : 1,
            "stackgroups": [{"title": project+"_StackGroup", "relation": "view"}]
          },
          {
            "title": stack+"zy",
            "dimension": str(tuple([shape[2],shape[1],shape[0]])),
            "mirrors": [{
              "fileextension": "raw",
              "position": 0,
              "tile_source_type": 14,
              "tile_height":tile_dim[0],
              "tile_width":tile_dim[1],
              "title": stack+"_tiles",
              "url": url + "2_1_0"
            }],
            "resolution": str(tuple([res[2],res[1],res[0]])),
            "translation": str(tuple([translation[2],translation[1],translation[0]])),
            "downsample_factors": ["(1,1,1)", "(2,2,2)", "(4,4,4)"],
            "orientation" : 2,
            "stackgroups": [{"title": project+"_StackGroup", "relation": "view"}]
          }]
        }
      }]
      
    else:
      stack_file = [{
        "project": {
          "title": project,
          "stacks": [{
            "title": stack+"xy",
            "dimension": str(tuple(shape)),
            "mirrors": [{
              "fileextension": "raw",
              "position": 0,
              "tile_source_type": 14,
              "tile_height":tile_dim[0],
              "tile_width":tile_dim[1],
              "title": stack+"_tiles",
              "url": url + "0_1_2"
            }],
            "resolution": str(tuple(res)),
            "translation": str(translation),
            "downsample_factors": ["(1,1,1)", "(2,2,2)", "(4,4,4)"],
            "orientation" : 0,
            "stackgroups": [{"title": project+"_StackGroup", "relation": "channel"}]
          }]
        }
      }]
                                                              
    #save json to local                    
    json_fpath = os.path.join("./", "data.json")       
    with open(json_fpath, 'w') as f:
        json.dump(stack_file, f)
    
    #open docker container and import json
    client = docker.from_env()
    container = client.containers.get(container_id)
    copy_string = json_fpath + " " + container_id + ":" + "/home/django/projects/data.json"
    os.system("docker cp " + str(copy_string))
    container.exec_run('python3 manage.py catmaid_import_projects --input data.json --permission user:admin:can_import user:admin:can_annotate')

    #clean-up
    container.exec_run('rm input data.json')
    client.close()
    os.remove(json_fpath)

    return stack_file