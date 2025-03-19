import numpy as np

def lut_preprocess_array(arr, max_int):
    arr_max = arr.max()
    if not max_int:
        max_int=arr.max()
    lut = np.empty(int(arr_max + max_int), dtype="uint8")
    lut[max_int:] = 255
    lut[:max_int] = np.round(np.arange(max_int) * (255 / max_int))
    return lut[arr]
    
    
    
def lut_preprocess_array(arr, max_int):
    dtype =  str(arr.dtype)
    arr_max = int(arr.max())
    if not max_int:
        max_int=arr_max
    max_int=int(max_int)
    lut = np.empty(int(arr_max + max_int), dtype="uint8")
    lut[max_int:] = 255
    lut[:max_int] = np.round(np.arange(max_int) * (255 / max_int))
    return lut[arr.astype('int16')].astype(dtype)