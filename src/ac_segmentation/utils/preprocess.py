import numpy as np
import itertools

def lut_preprocess_array(arr, max_int):
    arr_max = arr.max()
    if not max_int:
        max_int=arr.max()
    lut = np.empty(int(arr_max + max_int), dtype="uint8")
    lut[max_int:] = 255
    lut[:max_int] = np.round(np.arange(max_int) * (255 / max_int))
    return lut[arr]
    
    
    
def lut_preprocess_array_minmax(arr, min_int=None, max_int=None):
    dtype =  str(arr.dtype)
    arr_max = int(arr.max())
    if not max_int:
        max_int=arr_max
    max_int=int(max_int)
    lut = np.empty(int(arr_max + max_int), dtype="uint8")
    if min_int:
        lut[:min_int] = 0
    lut[max_int:] = 255
    lut[:max_int] = np.round(np.arange(max_int) * (255 / max_int))
    if 'int' not in str(arr.dtype):
        arr = arr.astype('int16')
    return lut[arr].astype(dtype)
    
    
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
    s = np.array(start[-3:])
    e = np.array(end[-3:])
    
    new_start.append(list(s))
    new_start.append(list(s+np.array([32,0,0])))
    new_start.append(list(s+np.array([0,32,0])))
    new_start.append(list(s+np.array([0,0,32])))

    new_end.append(list(e))
    new_end.append(list(e+np.array([32,0,0])))
    new_end.append(list(e+np.array([0,32,0])))
    new_end.append(list(e+np.array([0,0,32])))

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