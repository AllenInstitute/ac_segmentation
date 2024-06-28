import numpy
import torch

try:
    import zarr
except ImportError:
    pass

import ac_segmentation
import ac_segmentation.neurotorch.datasets.datatypes
import ac_segmentation.neurotorch.datasets.dataset
import ac_segmentation.neurotorch.nets.RSUNet
import ac_segmentation.neurotorch.core.predictor
from ac_segmentation.utils.tensorstore import open_ZarrTensor
from ac_segmentation.utils.preprocess import lut_preprocess_array


Predictor = ac_segmentation.neurotorch.core.predictor.Predictor
Vector = ac_segmentation.neurotorch.datasets.datatypes.Vector
BoundingBox = ac_segmentation.neurotorch.datasets.datatypes.BoundingBox
TSArray = ac_segmentation.neurotorch.datasets.dataset.TSArray
Array = ac_segmentation.neurotorch.datasets.dataset.Array
RSUNet = ac_segmentation.neurotorch.nets.RSUNet.RSUNet
np = numpy

ONE_GiB = 1_000_000_000


def predict_array(
        weights_file, arr,
        iter_size=BoundingBox(Vector(0, 0, 0), Vector(64, 64, 64)),
        stride=Vector(32, 32, 32),
        batch_size=80, gpu_device=None):
    inarr = Array(arr, iteration_size=iter_size, stride=stride)
    outarr = Array(
        -np.inf * np.ones(
            inarr.getBoundingBox().getNumpyDim(), dtype=np.float32),
        iteration_size=iter_size, stride=stride)
    net = ac_segmentation.neurotorch.nets.RSUNet.RSUNet()
    predictor = Predictor(net, weights_file, gpu_device=gpu_device)
    predictor.run(inarr, outarr, batch_size=batch_size)

    # prob_map = 1/(1+np.exp(-outarr.getArray()))
    prob_map = torch.special.expit(
        torch.from_numpy(outarr.getArray())
    ).numpy()
    return prob_map


# TODO predict_arr_chunked function


def predict_zarr(zarr_loc, weights_file, level=0,
                 max_intensity=30000, **kwargs):
    z = zarr.load(zarr_loc)
    ds = z[level]
    data = numpy.transpose(ds[0, 0, ...])
    data = lut_preprocess_array(data, max_intensity)

    prob_arr = predict_array(weights_file, data, **kwargs)
    return numpy.transpose(prob_arr)


def predict_zarr_ts(zarr_loc, weights_file, level=0,
                    max_intensity=30000, bytes_limit=(5 * ONE_GiB),
                    iter_size=BoundingBox(Vector(0, 0, 0), Vector(64, 64, 64)),
                    stride=Vector(32, 32, 32),
                    batch_size=80, gpu_device=None):
    try:
      in_ts = open_ZarrTensor(f"{zarr_loc}/{level}", bytes_limit=bytes_limit)
    except:
      in_ts = open_ZarrTensor(zarr_loc, bytes_limit=bytes_limit)
    
    in_ts = numpy.transpose(in_ts[0, 0, ...])
    in_arr = TSArray(in_ts, iteration_size=iter_size, stride=stride)
    out_arr = Array(
        -np.inf * np.ones(in_ts.shape,
                          dtype=np.float32),
        iteration_size=iter_size, stride=stride)
    net = ac_segmentation.neurotorch.nets.RSUNet.RSUNet()
    predictor = Predictor(net, weights_file, gpu_device=gpu_device)
    predictor.run(
        in_arr, out_arr, batch_size=batch_size, max_pix=max_intensity)

    # prob_map = 1/(1+np.exp(-outarr.getArray()))
    prob_map = torch.special.expit(
        torch.from_numpy(
            out_arr.getArray())
    ).numpy()
    return prob_map.transpose()
