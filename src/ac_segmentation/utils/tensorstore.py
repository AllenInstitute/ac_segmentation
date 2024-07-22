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

class AWS_Parameters:
    entries: dict[int, tuple[str, str]]
    temp_dir: TemporaryDirectory[str]
    credentials_file_path: Path

    @classmethod
    @lru_cache
    def singleton(cls) -> "Self":
        return cls()
        
    def __init__(self, profile=None, region=None, endpoint_url=None):
        self.entries = {}
        self.temp_dir = TemporaryDirectory()
        self.credentials_file_path = Path(self.temp_dir.name) / "aws_credentials"
        self.credentials_file_path.touch()

        #create session
        session = boto3.Session(profile_name=profile, region_name=region)
        if endpoint_url:
            session.endpoint_url=endpoint_url
        self.profile=session.profile_name
        self.region=session.region_name

    def _dump_credentials(self) -> None:
        self.credentials_file_path.write_text(
            "\n".join(
                [
                    f"[{self.profile}]\naws_access_key_id = {access_key_id}\naws_secret_access_key = {secret_access_key}\n"
                    for key_hash, (
                        access_key_id,
                        secret_access_key,
                    ) in self.entries.items()
                ]
            )
        )

    def add_credentials(self, access_key_id: str, secret_access_key: str) -> dict[str, str]:
        key_tuple = (access_key_id, secret_access_key)
        key_hash = hash(key_tuple)
        self.entries[key_hash] = key_tuple
        self._dump_credentials()
        self.credential_file = {
            "profile": f"profile-{key_hash}",
            "filename": str(self.credentials_file_path),
            "metadata_endpoint": "",
        }

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



def create_tensor(fpath, arr_shape, driver='zarr', store='file', dtype='float32', fill_value=-np.inf, 
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
            kvstore.update({"aws_region": AWS_param.region})
            if hasattr(AWS_param, "endpoint_url"):
                kvstore.update({"endpoint": AWS_param.endpoint_url})
            cred = {"aws_credentials":{"profile": AWS_param.profile}}
            if hasattr(AWS_param, "credential_file"):
                cred = {"aws_credentials":{"profile": AWS_param.profile, 
                                           "filename": AWS_param.credential_file['filename']}}
            kvstore.update(cred)

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

    return out_arr

def open_tensor(fpath, driver='zarr', store='file', AWS_param=None, bytes_limit= 100_000_000):
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
            kvstore.update({"aws_region": AWS_param.region})
            if hasattr(AWS_param, "endpoint_url"):
                kvstore.update({"endpoint": AWS_param.endpoint_url})
            cred = {"aws_credentials":{"profile": AWS_param.profile}}
            if hasattr(AWS_param, "credential_file"):
                cred = {"aws_credentials":{"profile": AWS_param.profile, 
                                           "filename": AWS_param.credential_file['filename']}}
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
        