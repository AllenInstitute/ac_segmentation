import pandas as pd
import numpy as np
import zarr
import docker
import tensorstore as ts
import os
import json
import boto3
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing_extensions import Self

def split_s3_path(s3_path):
    if 'https' in s3_path:
        path_parts=s3_path.replace("https://","").split("/")
        bucket=path_parts.pop(0).split(".s3")[0]
        key="/".join(path_parts)
    else:
        path_parts=s3_path.replace("s3://","").split("/")
        bucket=path_parts.pop(0)
        key="/".join(path_parts)
    return bucket, key

class AwsConfig:
    def __init__(self, 
                 aws_access_key_id: Optional[str] = None, 
                 aws_secret_access_key: Optional[str] = None, 
                 region_name: Optional[str] = None, 
                 endpoint_url: Optional[str] = None, 
                 profile: str = 'default'):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.profile = profile
        self.endpoint_url = endpoint_url

    def __repr__(self):
        return (f"AwsConfig(aws_access_key_id={self.aws_access_key_id}, "
                f"aws_secret_access_key={self.aws_secret_access_key}, "
                f"region_name={self.region_name}, "
                f"endpoint_url={self.endpoint_url}, "
                f"profile={self.profile})")

def load_aws_config(profile: str = 'default', 
                    endpoint_url: Optional[str] = None, 
                    default_region: str = 'us-west-1') -> AwsConfig:
    home = os.path.expanduser("~")
    credentials_path = os.path.join(home, ".aws", "credentials")
    config_path = os.path.join(home, ".aws", "config")

    config = configparser.ConfigParser()
    
    # Load environment variables as fallback
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region_name = os.getenv('AWS_REGION')

    # Load credentials
    if os.path.exists(credentials_path):
        config.read(credentials_path)
        if profile in config:
            aws_access_key_id = config.get(profile, "aws_access_key_id", fallback=aws_access_key_id)
            aws_secret_access_key = config.get(profile, "aws_secret_access_key", fallback=aws_secret_access_key)

    # Load config
    if os.path.exists(config_path):
        config.read(config_path)
        if profile in config:
            region_name = config.get(profile, "region", fallback=region_name)
    
    # Ensure region_name is set to default if not found
    if region_name is None:
        region_name = default_region

    return AwsConfig(aws_access_key_id, aws_secret_access_key, region_name, endpoint_url, profile)

def add_aws_profile(profile: str, 
                    aws_access_key_id: str, 
                    aws_secret_access_key: str) -> None:
    home = os.path.expanduser("~")
    credentials_path = os.path.join(home, ".aws", "credentials")
    
    config = configparser.ConfigParser()

    # Read existing profiles
    if os.path.exists(credentials_path):
        config.read(credentials_path)

    # Check if profile already exists
    if profile in config:
        raise ValueError(f"Profile '{profile}' already exists in {credentials_path}.")

    # Add new profile
    config[profile] = {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key
    }

    # Write the updated config back to the file
    with open(credentials_path, 'w') as configfile:
        config.write(configfile)

    print(f"Profile '{profile}' added successfully to {credentials_path}.")


def open_tensor(fpath, driver='zarr3', store='file', AWS_param=None, bytes_limit= 100_000_000):
    """Open a tensorstore object.
       driver: Type of file, including zarr, n5, precomputed
       store: Type of source, including file, s3
       AWS_client: Only applicable to s3 store
    """
    kvstore = {"driver": store,"path": fpath}
    if store == 's3':
        bucket,path = split_s3_path(fpath)
        kvstore = {"driver": "s3","bucket": bucket ,"path": path}
        if AWS_param:
            kvstore.update({"aws_region": AWS_param.region_name})
            if AWS_param.endpoint_url:
                kvstore.update({"endpoint": AWS_param.endpoint_url})
            cred = {"aws_credentials":{"profile": AWS_param.profile}}
            kvstore.update(cred)
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
             \
         },
         'recheck_cached_data':
         'open',
     })

    return dataset_future.result()

def create_tensor(fpath, arr_shape, driver='zarr3', store='file', dtype='float32', fill_value=-np.inf, 
                       chunk_shape=[64, 64, 64], res=[1,1,1], scale=0, arr=None, AWS_param=None):
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
        bucket,path = split_s3_path(fpath)
        kvstore = {"driver": "s3","bucket": bucket ,"path": path}
        if AWS_param:
            kvstore.update({"aws_region": AWS_param.region_name})
            if AWS_param.endpoint_url:
                kvstore.update({"endpoint": AWS_param.endpoint_url})
            cred = {"aws_credentials":{"profile": AWS_param.profile}}
            kvstore.update(cred)

    if driver in ['zarr','zarr3','n5']:
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

    return out_arr


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
    
def zarr_to_precomputed(zarr_path, out_path, store='file', chunks=(64,64,64), cutout=None, scales=6, AWS_param=None):
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
        pre_comp = create_tensor(out_path, arr_shape=arr.shape, dtype=arr.dtype, store=store, driver='neuroglancer_precomputed', AWS_param=AWS_param, scale=scale)
        pre_comp.write(arr).result()
