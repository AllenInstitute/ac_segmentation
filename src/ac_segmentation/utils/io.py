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
    with gzip.open(fn, "wb") as f:
        numpy.save(f, arr)


def read_gzip_array(fn, preprocess_func=None):
    with gzip.open(fn, "rb") as f:
        x = numpy.load(f)
    if preprocess_func:
        x = preprocess_func(x)
    return x


# FIXME CL code does not preserve ids
def write_cv_skels_iter_tar(tar_fn, skels):
    with tarfile.open(tar_fn, mode="w:gz") as t:
        for skid, skel in enumerate(skels):
            bio = io.BytesIO(skel.to_swc().encode())
            info = tarfile.TarInfo(name=f"{skid}.swc")
            info.size = len(bio.getbuffer())
            t.addfile(tarinfo=info, fileobj=bio)
            
    
def write_cv_skels_tar(tar_fn, skels, mode='w:gz'):
    with tarfile.open(tar_fn, mode=mode) as t:
        id = 1
        for skel in skels:
            bio = BytesIO(skel.to_swc().encode())
            info = tarfile.TarInfo(name=f"{id}.swc")
            info.size = len(bio.getbuffer())
            t.addfile(tarinfo=info, fileobj=bio)
            id += 1
            
            
def read_swc_cv(swc, id=0):
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
