import concurrent.futures
import gzip
import io
import tarfile
import numpy
import boto3
from io import BytesIO
from ac_segmentation.utils.tensorstore import split_s3_path


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
