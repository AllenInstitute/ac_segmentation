import numpy as np

def lut_preprocess_array(arr, max_int):
    arr_max = arr.max()
    if not max_int:
        max_int=arr.max()
    lut = np.empty(arr_max + max_int, dtype="uint8")
    lut[max_int:] = 255
    lut[:max_int] = np.round(np.arange(max_int) * (255 / max_int))
    return lut[arr]
