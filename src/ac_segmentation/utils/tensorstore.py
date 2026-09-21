import configparser
from typing import Optional
import pandas as pd
import numpy as np
import tensorstore as ts
import os
import json
import boto3
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing_extensions import Self


def split_s3_path(s3_path):
    """Parse an S3 URI (s3:// or https://) into its bucket name and object key.

    Args:
        s3_path (str): S3 URI, either in s3://bucket/key or https://bucket.s3.../key form.
    """
    if 'https' in s3_path:
        path_parts=s3_path.replace("https://","").split("/")
        bucket=path_parts.pop(0).split(".s3")[0]
        key="/".join(path_parts)
    else:
        path_parts=s3_path.replace("s3://","").split("/")
        bucket=path_parts.pop(0)
        key="/".join(path_parts)
    return bucket, key

class AWS_Parameters:
    """Holds AWS session/profile/region/endpoint info and manages a temporary on-disk credentials file for tensorstore's S3 kvstore driver.

    Args:
        entries (dict[int, tuple[str, str]]): Registered access-key/secret-key pairs, keyed by a hash of the pair.
        temp_dir (TemporaryDirectory[str]): Temporary directory backing the generated credentials file.
        credentials_file_path (Path): Path to the generated AWS credentials file.
    """
    entries: dict[int, tuple[str, str]]
    temp_dir: TemporaryDirectory[str]
    credentials_file_path: Path
    @classmethod
    @lru_cache
    def singleton(cls) -> "Self":
        """Return a process-wide cached singleton instance of AWS_Parameters.

        Args:
            cls (type[AWS_Parameters]): Class this method is bound to, used to instantiate the singleton.
        """
        return cls()
        
    def __init__(self, profile=None, region=None, endpoint_url=None):
        """Create a boto3 session for the given profile/region, set up a temporary credentials file, and record the resolved profile, region, and endpoint.

        Args:
            profile (str | None): Named AWS credentials profile to use for the session.
            region (str | None): AWS region for the session.
            endpoint_url (str | None): Custom S3-compatible endpoint URL to associate with this instance.
        """
        self.entries = {}
        self.temp_dir = TemporaryDirectory()
        self.credentials_file_path = Path(self.temp_dir.name) / "aws_credentials"
        self.credentials_file_path.touch()
        #create session
        session = boto3.Session(profile_name=profile, region_name=region)
        if endpoint_url:
            self.endpoint_url=endpoint_url
        self.profile=session.profile_name
        self.region=session.region_name
    def _dump_credentials(self) -> None:
        """Write all registered access-key/secret-key entries to the temporary AWS credentials file in INI format.

        Args:
            self (AWS_Parameters): Instance whose self.entries are serialized.
        """
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
        """Register an access key / secret key pair, persist it to the temporary credentials file, and record the resulting profile/file info for use by tensorstore's S3 kvstore.

        Args:
            access_key_id (str): AWS access key ID to register.
            secret_access_key (str): AWS secret access key to register.
        """
        key_tuple = (access_key_id, secret_access_key)
        key_hash = hash(key_tuple)
        self.entries[key_hash] = key_tuple
        self._dump_credentials()
        self.credential_file = {
            "profile": f"profile-{key_hash}",
            "filename": str(self.credentials_file_path),
            "metadata_endpoint": "",
        }


def create_kvstore(fpath, store, AWS_param=None):
    """Build a tensorstore kvstore configuration for a local file path or an S3 bucket/path, optionally attaching AWS region, endpoint, and credential information.

    Args:
        fpath (str): Path to the tensorstore file, or an s3:// URL when store='s3'.
        store (str): Kvstore driver to use, e.g. 'file' or 's3'.
        AWS_param (AWS_Parameters | None): AWS credentials/region/endpoint to attach when store='s3'.
    """
    kvstore = {"driver": store, "path": fpath}
    
    if store == 's3':
        # Parse the S3 URL into bucket and path
        bucket, path = split_s3_path(fpath)
        kvstore = {"driver": "s3", "bucket": bucket, "path": path}
        
        if AWS_param:
            kvstore.update({"aws_region": AWS_param.region})
            if hasattr(AWS_param, "endpoint_url"):
                kvstore.update({"endpoint": AWS_param.endpoint_url})
            
            # Handle credentials
            cred = {"aws_credentials": {"profile": AWS_param.profile}}
            if hasattr(AWS_param, "credential_file"):
                cred = {"aws_credentials": {
                    "profile": AWS_param.profile,
                    "filename": AWS_param.credential_file['filename']
                }}
            kvstore.update(cred)
    
    return kvstore
    
    
def open_tensor(fpath=None, kvstore=None, driver='zarr', bytes_limit=100_000_000):
    """Open an existing tensorstore dataset, building its kvstore from a local/S3 path if one isn't provided, and falling back from 'zarr' to 'zarr3' if needed.

    Args:
        fpath (str | None): Path to the tensorstore file or S3 URL, used to build a kvstore if kvstore is not given.
        kvstore (dict | None): Pre-constructed kvstore configuration; overrides fpath if provided.
        driver (str): Tensorstore driver to use, e.g. 'zarr', 'n5', or 'precomputed'.
        bytes_limit (int): In-memory cache size limit, in bytes.
    """
    # If kvstore is not provided, create it from fpath
    if kvstore is None:
        kvstore = create_kvstore(fpath, store='file', AWS_param=None)

    # Check if zarr v3
    if 'zarr' in driver:
        # Load the tensorstore array with cache configuration
        try:
            dataset_future = ts.open({
                'driver': 'zarr',
                'kvstore': kvstore,
                'context': {
                    'cache_pool': {
                        'total_bytes_limit': bytes_limit
                    }
                },
                'recheck_cached_data': 'open',
            })
            return dataset_future.result()
    
        except:
            dataset_future = ts.open({
                'driver': 'zarr3',
                'kvstore': kvstore,
                'context': {
                    'cache_pool': {
                        'total_bytes_limit': bytes_limit
                    }
                },
                'recheck_cached_data': 'open',
            })
            return dataset_future.result()
            
    else:
         dataset_future = ts.open({
                'driver': driver,
                'kvstore': kvstore,
                'context': {
                    'cache_pool': {
                        'total_bytes_limit': bytes_limit
                    }
                },
                'recheck_cached_data': 'open',
            })
         return dataset_future.result()



def create_tensor(arr_shape, fpath=None, kvstore=None, driver='zarr3', dtype='float32', fill_value=0, 
                       chunk_shape=[64, 64, 64], shard_shape=None, res=[1,1,1], scale=0, codecs=None, index_codecs=None, sharded=False, shard_factor=4):
    """Create a new tensorstore array with the given shape, dtype, and chunking, supporting the 'zarr', 'zarr3' (optionally sharded), 'n5', and 'neuroglancer_precomputed' drivers.

    Args:
        arr_shape (Sequence[int]): Shape of the array to create.
        fpath (str | None): Path to create the tensorstore file/S3 URL at, used to build a kvstore if kvstore is not given.
        kvstore (dict | None): Pre-constructed kvstore configuration; overrides fpath if provided.
        driver (str): Tensorstore driver to create, one of 'zarr', 'zarr3', 'n5', or 'neuroglancer_precomputed'.
        dtype (str): Data type of the new array.
        fill_value (int | float): Fill value for uninitialized chunks (forced to 0 for integer dtypes).
        chunk_shape (list[int]): Chunk shape for the array, or inner chunk shape when sharded.
        shard_shape (list[int] | None): Outer shard shape for zarr3 sharded arrays; computed from chunk_shape and shard_factor if not given.
        res (list[int]): Voxel resolution, used by the 'neuroglancer_precomputed' driver's scale metadata.
        scale (int): Scale index, used to name the 'neuroglancer_precomputed' driver's scale key.
        codecs (dict | None): Codec configuration (e.g. compressor) applied to the array's data.
        index_codecs (dict | None): Codec configuration applied to the shard index, for sharded zarr3 arrays.
        sharded (bool): Whether to create a zarr3 array with sharding enabled.
        shard_factor (int): Multiplier applied to chunk_shape's spatial dims to derive shard_shape when sharding is enabled and shard_shape isn't given.
    """
    if 'int' in str(dtype):
        fill_value=0

     # If kvstore is not provided, create it from fpath
    if kvstore is None:
        kvstore = create_kvstore(fpath, store='file', AWS_param=None)

    if driver == 'zarr':
        out_arr = ts.open({
            "driver": "zarr",
            "kvstore": kvstore,
            "key_encoding": ".",
            "metadata": {
                "shape": list(arr_shape),
                "chunks": chunk_shape,
                "order": "C",
                "compressor": codecs
            },
            "dtype":dtype
        },
        fill_value=fill_value,
        create=True,  # this is what makes it a new one
        delete_existing=False  # optional: overwrite any existing array
        ).result()

    if driver == 'zarr3':
        meta = {
            "driver": "zarr3",
            "kvstore": kvstore,
            "metadata": {
                "shape": list(arr_shape),
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk_shape}},
                "data_type": dtype,
                "codecs": []
            }
        }

        if codecs:
            meta['metadata']['codecs'] = [codecs]
        if index_codecs:
            meta['metadata']['index_codecs'] = [index_codecs]

        if sharded == True:
            if not shard_shape:
                shard_shape = list(np.array(chunk_shape[:-3] + [x * (shard_factor) for x in chunk_shape[-3:]]))
            meta['metadata']['chunk_grid']['configuration']['chunk_shape']=shard_shape
            shard_meta = {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": chunk_shape,
                        "codecs": [],
                        "index_codecs": [],
                        "index_location": "end"
                            }
                        }
            meta['metadata']['codecs'] = [shard_meta]

            if codecs:
                meta['metadata']['codecs'][0]['configuration']['codecs'] = [codecs]
            if index_codecs:
                meta['metadata']['codecs'][0]['configuration']['index_codecs'] = [index_codecs]
            
        out_arr =ts.open(meta,
        fill_value = 0,
        create=True,  
        delete_existing=False 
        ).result()

    if driver == 'n5':
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

    return out_arr




create_EmptyTensor = create_tensor  
open_ZarrTensor = open_tensor
    
def zarr_to_n5(zarr_path, out_path, chunks=(64,64,64), cutout=None):
    """Read a Zarr tensorstore array, optionally a sub-region, and write it out as an N5 array on disk.

    Args:
        zarr_path (str): Path to the source Zarr array.
        out_path (str): Directory in which the output N5 store is created.
        chunks (tuple[int, int, int]): Chunk shape for the output N5 array.
        cutout (Sequence[int] | None): Optional (x1, x2, y1, y2, z1, z2) bounding box restricting which region is copied.
    """
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
    """Convert a multi-scale Zarr pyramid into a Neuroglancer precomputed volume, copying each scale level and reading its resolution from the Zarr multiscale metadata.

    Args:
        zarr_path (str): Base path to the source Zarr array; each scale level is read from zarr_path + str(scale).
        out_path (str): Path where the output precomputed volume is created.
        store (str): Kvstore driver for the output ('file', 's3', etc.).
        chunks (tuple[int, int, int]): Chunk shape used when creating each output scale's tensor.
        cutout (Sequence[int] | None): Optional (x1, x2, y1, y2, z1, z2) bounding box restricting which region is copied at scale 0; halved at each subsequent scale.
        scales (int): Number of scale levels to convert, starting at 0.
        AWS_param (AWS_Parameters | None): AWS credentials/region/endpoint used when creating the output tensor.
    """
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
