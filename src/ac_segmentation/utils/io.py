import concurrent.futures
import gzip
import io
import itertools
import tarfile
import numpy
import boto3
from io import BytesIO
import fastremap
from ac_segmentation.utils.tensorstore import split_s3_path
import navis
import tensorstore as ts


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
            info = tarfile.TarInfo(name=f"{skel.id}.swc")
            info.size = len(bio.getbuffer())
            t.addfile(tarinfo=info, fileobj=bio)
            
            
def write_navis_skels_tar(tar_fn, skels, mode='w:gz', swcname=False):
    with tarfile.open(tar_fn, mode=mode) as t:
        for sk in skels:
            id = sk.id
            if swcname:
                id = sk.swcname
            if 'label' not in sk.nodes:
                sk.nodes.insert(1, 'label', list(np.zeros(len(sk.nodes))))
            sk = sk.nodes[['node_id', 'label','x','y','z','radius','parent_id']].values.tolist()
            sk = '\n'.join(str(x)[1:-1] for x in sk).replace(",", "")
            bio = io.BytesIO(sk.encode())
            info = tarfile.TarInfo(name=f"{id}.swc")
            info.size = len(bio.getbuffer())
            t.addfile(tarinfo=info, fileobj=bio)
            
            
def process_swc_file(swc_data, swcname, file_id):
    neuron = navis.io.read_swc(f=swc_data, swcname=swcname)
    neuron.id = file_id
    return neuron

def read_navis_neurons_tar(tar_fn, concurrency=10, preprocess_func=None, uuid=True):
    preprocess_func = ((lambda x: x) if preprocess_func is None else preprocess_func)
    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as e:
        futs = []
        with tarfile.open(tar_fn, "r:gz") as t:
            for m in t.getmembers():
                # Extract the SWC file contents
                swc_b = t.extractfile(m).read()
                file_id = m.name.split('.')[0]
                try:
                    file_id = int(file_id)
                except:
                    pass
                if uuid:
                    futs.append(e.submit(navis.io.read_swc,f=swc_b.decode(),swcname=file_id))
                else:
                    futs.append(e.submit(process_swc_file, swc_b.decode(), file_id, file_id))
                
        neurons = [preprocess_func(fut.result()) for fut in concurrent.futures.as_completed(futs)]
        navis_neurons = navis.NeuronList([n for n in neurons if not n is None])
    return navis_neurons
            
            
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
        
def create_chunked_dims(arr_shape, chunk_size):
    # Ensure chunk_size is appropriate for the arr_shape length
    if len(arr_shape) != len(chunk_size):
        raise ValueError("arr_shape and chunk_size must have the same number of dimensions")

    # Get indexing combinations
    start_indices = []
    end_indices = []

    for dim_size, chunk in zip(arr_shape, chunk_size):
        # Generate the start indices for each chunk
        start_indices.append(list(range(0, dim_size, chunk)))
        # Generate the end indices for each chunk, ensuring not to exceed the array size
        end_indices.append([min(dim_size, start + chunk) for start in start_indices[-1]])

    # Create all combinations of start and end indices across all dimensions
    start = [list(item) for item in itertools.product(*start_indices)]
    end = [list(item) for item in itertools.product(*end_indices)]

    return start, end

def create_overlap_chunks(start, end, overlap=32):
    new_start = []
    new_end = []
    s = np.array(start)
    e = np.array(end)

    arrays = [np.array([overlap,0,0]),np.array([0,overlap,0]),np.array([0,0,overlap]),np.array([overlap,0,overlap]),
              np.array([0,overlap,overlap]),np.array([overlap,overlap,0]), np.array([overlap,overlap,overlap])]

    new_start.append(list(s))
    new_end.append(list(e))
    for arr in arrays:
        new_start.append(list(s+arr))
        new_end.append(list(e+arr))
        
    return [new_start,new_end]
        

def remap_arr(in_arr, mappings, chunk_size=[1000,1000,1000]):
    if isinstance(in_arr, numpy.ndarray):
        out_arr = fastremap.remap(in_arr, mappings, preserve_missing_labels=True)
        return out_arr
    if isinstance(in_arr, ts.TensorStore):
        start,end = create_chunked_dims(arr_shape=in_arr.shape, chunk_size=chunk_size)
        for s, e in zip(start,end):
            arr = in_arr[s[0]:e[0],s[1]:e[1],s[2]:e[2]].read().result()
            out_arr = fastremap.remap(arr, mappings, preserve_missing_labels=True)
            in_arr[s[0]:e[0],s[1]:e[1],s[2]:e[2]].write(out_arr).result()
