import concurrent.futures
import gzip
import io
import tarfile
import numpy
import boto3
from io import BytesIO
from ac_segmentation.utils.tensorstore import split_s3_path
import os
from cloudvolume import Skeleton
import re
import concurrent
from concurrent.futures import ThreadPoolExecutor
import uuid


def gzip_array(fn, arr):
    """Save a NumPy array to disk as a gzip-compressed .npy file.

    Args:
        fn (str): Path to write the gzip-compressed array to.
        arr (np.ndarray): Array to save.

    Example:
        >>> import numpy as np
        >>> gzip_array('out.npy.gz', np.zeros((10, 10)))  # doctest: +SKIP
    """
    with gzip.open(fn, "wb") as f:
        numpy.save(f, arr)


def read_gzip_array(fn, preprocess_func=None):
    """Load a NumPy array from a gzip-compressed .npy file, optionally applying a preprocessing function to it.

    Args:
        fn (str): Path to the gzip-compressed array file.
        preprocess_func (Callable[[np.ndarray], np.ndarray] | None): Optional function applied to the loaded array before returning it.

    Example:
        >>> arr = read_gzip_array('out.npy.gz')  # doctest: +SKIP
    """
    with gzip.open(fn, "rb") as f:
        x = numpy.load(f)
    if preprocess_func:
        x = preprocess_func(x)
    return x


# FIXME CL code does not preserve ids
def write_cv_skels_iter_tar(tar_fn, skels):
    """Write an iterable of cloudvolume skeletons to a tar.gz archive as individual SWC files, named by their position in the iterable.

    Args:
        tar_fn (str): Path to the output tar.gz archive.
        skels (Iterable[cloudvolume.Skeleton]): Skeletons to write, in iteration order. Note that original skeleton IDs are not preserved; files are named by enumeration index.
    """
    with tarfile.open(tar_fn, mode="w:gz") as t:
        for skid, skel in enumerate(skels):
            bio = io.BytesIO(skel.to_swc().encode())
            info = tarfile.TarInfo(name=f"{skid}.swc")
            info.size = len(bio.getbuffer())
            t.addfile(tarinfo=info, fileobj=bio)
            
    
def write_cv_skels_tar(tar_fn, skels, mode='w:gz'):
    """Write a collection of cloudvolume skeletons to a tar archive as individual SWC files, numbered sequentially starting at 1.

    Args:
        tar_fn (str): Path to the output tar archive.
        skels (Iterable[cloudvolume.Skeleton]): Skeletons to write, in iteration order.
        mode (str): tarfile open mode, e.g. 'w:gz' for gzip-compressed or 'w' for uncompressed.
    """
    with tarfile.open(tar_fn, mode=mode) as t:
        id = 1
        for skel in skels:
            bio = BytesIO(skel.to_swc().encode())
            info = tarfile.TarInfo(name=f"{id}.swc")
            info.size = len(bio.getbuffer())
            t.addfile(tarinfo=info, fileobj=bio)
            id += 1
            
            
def read_swc_cv(swc, id=0):
    """Parse an SWC string into a cloudvolume Skeleton, coercing the node ID, type, and parent ID columns to integers.

    Args:
        swc (str): Raw SWC-formatted text to parse.
        id (int): ID to assign to the resulting skeleton.
    """
    fixed_lines = []
    for line in swc.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            fixed_lines.append(line)
            continue
        parts = line.split()
        # Columns 0 (id), 1 (type), 6 (parent_id) should be ints
        for i in [0, 1, 6]:
            if i < len(parts):
                try:
                    parts[i] = str(int(float(parts[i])))
                except ValueError:
                    pass
        fixed_lines.append(' '.join(parts))
    fixed_swc = '\n'.join(fixed_lines)
    
    skel = Skeleton.from_swc(fixed_swc)
    skel.id = id
    return skel

def read_cv_neurons_tar(tar_fn, n_workers=10, preprocess_func=None):
    """Read all SWC members from a tar archive in parallel and parse each into a cloudvolume Skeleton with a sequential ID.

    Args:
        tar_fn (str): Path to the tar archive containing SWC files.
        n_workers (int): Number of worker processes used to parse SWC files in parallel.
        preprocess_func (Callable[[cloudvolume.Skeleton], Any] | None): Optional function applied to each parsed skeleton before it is returned.
    """
    preprocess_func = ((lambda x: x) if preprocess_func is None else preprocess_func)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as e:
        futs = []
        with tarfile.open(tar_fn, "r:*") as t:
            id = 1
            for m in t.getmembers():
                swc_b = t.extractfile(m).read()
                futs.append(e.submit(read_swc_cv, swc_b.decode(), id))
                id += 1
        cv_neurons = [preprocess_func(fut.result()) for fut in concurrent.futures.as_completed(futs)]
    return cv_neurons
    
    
def cv_to_navis(skels, tag=None):
    """Convert cloudvolume skeletons into a navis NeuronList via each skeleton's SWC representation.

    Args:
        skels (Iterable[cloudvolume.Skeleton]): Skeletons to convert.
        tag (str | None): If provided, assigned as the .name attribute of every resulting neuron.
    """
    out_sk = navis.NeuronList(None)
    try:
        for sk in skels:
            sk = navis.TreeNeuron(sk.to_swc())
            if tag:
                sk.name = tag
            out_sk.append(sk)
    except:
        out_sk.append(navis.NeuronList(skels.to_swc()))

    return out_sk
            
            
def upload_to_ceph(arr, out_file, profile=None, endpoint=None, aws_access_key=None, aws_secret_key=None, region='us-east-1'):
    """Gzip-compress a NumPy array in memory and upload it to an S3-compatible (Ceph) bucket using the given credentials or profile.

    Args:
        arr (np.ndarray): Array to gzip-compress and upload.
        out_file (str): Destination S3 URI (bucket/key) to upload to.
        profile (str | None): Named AWS credentials profile to use, if not passing explicit keys.
        endpoint (str | None): Custom S3-compatible endpoint URL.
        aws_access_key (str | None): AWS access key ID, if not using a profile or environment credentials.
        aws_secret_key (str | None): AWS secret access key paired with aws_access_key.
        region (str): AWS region used for the S3 session.
    """
    try:
        # Gzip the NumPy array and write it to the buffer
        buffer = BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
            numpy.save(f, arr)  # Save the array as .npy in gzip format
        buffer.seek(0)
        
        # If AWS credentials are provided, use them
        if aws_access_key and aws_secret_key:
            session = boto3.Session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region  # Default region, you can modify this as needed
            )
        elif profile:
            session = boto3.Session(profile_name=profile)
        else:
            # Default to environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
            session = boto3.Session()

        # Create the S3 client with the session and endpoint
        client = session.client('s3', endpoint_url=endpoint)
        
        # Upload the gzipped data to the Ceph bucket
        bucket,key = split_s3_path(out_file)
        response = client.put_object(Bucket=bucket, Key=key, Body=buffer)

        # Optionally log or return the response from the upload
        print(f"Upload successful")
    except Exception as e:
        print(f"An error occurred during upload: {e}")
