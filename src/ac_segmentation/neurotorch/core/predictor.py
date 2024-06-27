import torch
from torch.autograd import Variable
import numpy as np
import joblib
from joblib import Parallel, delayed
from ac_segmentation.neurotorch.datasets.dataset import Data
from ac_segmentation.utils.preprocess import lut_preprocess_array
from skimage import exposure

import ac_segmentation.neurotorch.datasets.dataset
from ac_segmentation.neurotorch.datasets.datatypes import BoundingBox, Vector
Array = ac_segmentation.neurotorch.datasets.dataset.Array
TSArray = ac_segmentation.neurotorch.datasets.dataset.TSArray



class Predictor:
    """
    A predictor segments an input volume into an output volume
    """
    def __init__(self, net, checkpoint, gpu_device=None):
        self.setNet(net, gpu_device=gpu_device)
        self.loadCheckpoint(checkpoint)

    def setNet(self, net, gpu_device=None):
        self.device = torch.device("cuda:{}".format(gpu_device)
                                   if gpu_device is not None
                                   else "cpu")

        self.net = net.to(self.device).eval()

    def getNet(self):
        return self.net

    def loadCheckpoint(self, checkpoint):
        self.getNet().load_state_dict(torch.load(checkpoint, map_location=self.device))

    def run(self, input_volume, output_volume, batch_size=30, mini_batch_size=10, max_pix = 65535):
        
        self.setBatchSize(batch_size)
        with torch.no_grad():
            batch_list = [list(range(len(input_volume)))[i:i+self.getBatchSize()]
                          for i in range(0,
                                         len(input_volume),
                                         self.getBatchSize())]

            for batch_index in batch_list:
                keep = []
                batch = [input_volume[i] for i in batch_index]
                if isinstance(input_volume, TSArray):
                    batch = [Data(np.pad(i.array.result(), pad_width=i.pad_size, mode="constant"),i.bounding_box) for i in batch]

                if hasattr(input_volume, 'mask'):
                    masks = [input_volume.mask[i] for i in batch_index]
                    if isinstance(input_volume.mask, TSArray):
                        masks = [Data(i.array.result(), i.bounding_box) for i in masks]
                    
                for ind,data in enumerate(batch):
                    if hasattr(input_volume, 'li_thresh'):
                        batch[ind].array[batch[ind].array < input_volume.li_thresh] = 0

                    if hasattr(input_volume, 'rescale_perc'):
                        p1, p2 = np.percentile(batch[ind].array, input_volume.rescale_perc)
                        batch[ind].array = exposure.rescale_intensity(batch[ind].array, in_range=(p1, p2))

                    if hasattr(input_volume, 'mask'):
                        batch[ind].array = np.where(masks[ind].array, batch[ind].array, 0)
                        
                    if np.any(data.array) == True:
                        if max_pix != None:
                            batch[ind].array = lut_preprocess_array(batch[ind].array, max_pix)
                        keep.append(batch[ind])

                if isinstance(input_volume, TSArray):
                    self.run_batch(keep, output_volume, input_volume.shift, mini_batch_size=mini_batch_size)

                else:
                    self.run_batch(keep, output_volume, mini_batch_size=mini_batch_size)

    def getBatchSize(self):
        return self.batch_size

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

    def run_batch(self, batch, output_volume, in_shift=[0,0,0], mini_batch_size=10):
        bounding_boxes, arrays = self.toTorch(batch)
        inputs = Variable(arrays).float()

        data_list = []
        n_list = range(len(inputs))
        batch_list = [n_list[i:i + mini_batch_size] for i in range(0, len(n_list), mini_batch_size)]  
        for s_batch in batch_list:
            st,end = int(s_batch[0]), int(s_batch[-1])
            outputs = self.getNet()(inputs[st:end])
            data_list += self.toData(outputs, bounding_boxes[st:end])

        if hasattr(output_volume, 'tensor'):
            writes = []
            for data in data_list:
                writes.append(output_volume.blend(data, in_shift))
                
            for write in writes:
                write.result()  

        else:
            for data in data_list:
                v1 = data.bounding_box.edge1 - Vector(*in_shift[::-1])
                v2 = data.bounding_box.edge2 - Vector(*in_shift[::-1])
                data = Data(data.array, BoundingBox(v1,v2))
                output_volume.blend(data)

    def toArray(self, data):
        torch_data = data.getArray().astype(float)
        torch_data = torch_data.reshape(1, 1, *torch_data.shape)
        return torch_data

    def toTorch(self, batch):
        bounding_boxes = [data.getBoundingBox() for data in batch]
        arrays = [self.toArray(data) for data in batch]
        arrays = torch.from_numpy(np.concatenate(arrays, axis=0))
        arrays = arrays.to(self.device)

        return bounding_boxes, arrays

    def toData(self, tensor_list, bounding_boxes):
        tensor = torch.cat(tensor_list).data.cpu().numpy()
        batch = [Data(tensor[i][0], bounding_box)
                 for i, bounding_box in enumerate(bounding_boxes)]

        return batch