import numpy
import torch

try:
    import zarr
except ImportError:
    pass

import ac_segmentation.neurotorch.datasets.datatypes
import ac_segmentation.neurotorch.datasets.dataset
import ac_segmentation.neurotorch.nets.RSUNet
import ac_segmentation.neurotorch.core.predictor
from ac_segmentation.neurotorch.datasets.dataset import open_ZarrTensor


Predictor = ac_segmentation.neurotorch.core.predictor.Predictor
Vector = ac_segmentation.neurotorch.datasets.datatypes.Vector
BoundingBox = ac_segmentation.neurotorch.datasets.datatypes.BoundingBox
TSArray = ac_segmentation.neurotorch.datasets.dataset.TSArray
Array = ac_segmentation.neurotorch.datasets.dataset.Array
RSUNet = ac_segmentation.neurotorch.nets.RSUNet.RSUNet
np = numpy

ONE_GiB = 1_000_000_000


def lut_preprocess_array(arr, max_int):
    lut = np.empty(arr.max() + max_int, dtype="uint8")
    lut[max_int:] = 255
    lut[:max_int] = np.round(np.arange(max_int) * (255 / max_int))
    return lut[arr]


class HackyArray(Array):
    def set(self, data):
        # TODO
        """allow dropping data off of improperly shaped arrays"""
        data_bounding_box = data.getBoundingBox()
        data_array = data.getArray()

        if not data_bounding_box.isSubset(self.getBoundingBox()):
            raise ValueError("The bounding box must be a subset of the "
                             " volume")

        data_edge1, data_edge2 = data_bounding_box.getEdges()
        array_edge1, array_edge2 = self.getBoundingBox().getEdges()

        edge1 = data_edge1 - array_edge1
        edge2 = data_edge2 - array_edge1

        x1, y1, z1 = edge1.getComponents()
        # x2, y2, z2 = edge2.getComponents()
        
        x2, y2, z2 = (min(s, c) for s, c in zip(
            self.array.shape[::-1], edge2.getComponents()))

        self.array[z1:z2, y1:y2, x1:x2] = data_array[
            :z2 - z1, :y2 - y1, :x2 - x1
        ]

    def setIteration(self, iteration_size, stride):
        """
        TODO: hack -- allow reading chunks beyond bounds by implementing a real ceiling
        Sets the parameters for iterating through the dataset

        :param iteration_size: The size of each data sample in the volume
        :param stride: The displacement of each iteration
        """
        if not isinstance(iteration_size, BoundingBox):
            error_string = (
                "iteration_size must have type BoundingBox"
                " instead it has type {}")
            error_string = error_string.format(type(iteration_size))
            raise ValueError(error_string)

        if not isinstance(stride, Vector):
            raise ValueError("stride must have type Vector")

        if not iteration_size.isSubset(BoundingBox(
                Vector(0, 0, 0),
                self.getBoundingBox().getSize())):
            raise ValueError("iteration_size must be smaller than volume size")

        self.setIterationSize(iteration_size)
        self.setStride(stride)

        def ceil(x):
            return math.ceil(x)
            # return int(round(x))

        self.element_vec = Vector(*map(
            lambda L, l, s: ceil((L- l) / s + 1),
            self.getBoundingBox().getSize().getComponents(),
            self.iteration_size.getSize().getComponents(),
            self.stride.getComponents()))

        self.index = 0


def predict_array(
        weights_file, arr,
        iter_size=BoundingBox(Vector(0, 0, 0), Vector(64, 64, 64)),
        stride=Vector(32, 32, 32),
        batch_size=80, gpu_device=None):
    inarr = HackyArray(arr, iteration_size=iter_size, stride=stride)
    outarr = HackyArray(
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
    in_ts = open_ZarrTensor(f"{zarr_loc}/{level}", bytes_limit=bytes_limit)
    in_ts = numpy.transpose(in_ts[0, 0, ...])
    in_arr = TSArray(in_ts, iteration_size=iter_size, stride=stride)
    out_arr = HackyArray(
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
