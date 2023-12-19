import numpy as np


def lut_preprocess_array(arr, max_int):
    lut = np.empty(int(arr.max()+max_int), dtype="uint16")
    lut[max_int:] = 255
    lut[:max_int] = np.round(np.arange(max_int) * (255 / max_int))
    return lut[arr]
