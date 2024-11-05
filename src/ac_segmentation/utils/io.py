import concurrent.futures
import gzip
import io
import tarfile
import numpy
import boto3
from io import BytesIO



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
            
            
def upload_to_ceph(arr, out_file, bucket, profile, endpoint):

    # Gzip the NumPy array and write it to the buffer
    buffer = BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
        numpy.save(f, arr)
    # Reset buffer position to the beginning
    buffer.seek(0)
    
    # Create a boto3 session using the specified profile
    session = boto3.Session(profile_name=profile)
    client = session.client('s3', endpoint_url=endpoint)

    # Upload the gzipped data
    client.put_object(Bucket=bucket, Key=out_file, Body=buffer)