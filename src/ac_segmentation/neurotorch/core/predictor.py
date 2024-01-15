import torch
from torch.autograd import Variable
import numpy as np
import joblib
from joblib import Parallel, delayed
from ac_segmentation.neurotorch.datasets.dataset import Data
from ac_segmentation.preprocess import lut_preprocess_array


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

    def run(self, input_volume, output_volume, batch_size=30, max_pix = 30000):
        self.setBatchSize(batch_size)

        with torch.no_grad():
            batch_list = [list(range(len(input_volume)))[i:i+self.getBatchSize()]
                          for i in range(0,
                                         len(input_volume),
                                         self.getBatchSize())]

            for batch_index in batch_list:
                keep = []
                batch = [input_volume[i] for i in batch_index]

                if hasattr(input_volume, 'tensor'):
                    batch = [Data(np.pad(ind.array.result(), pad_width=ind.pad_size, mode="constant"),ind.bounding_box) for ind in batch]

                for ind,data in enumerate(batch):
                    if np.any(data.array) == True:
                        batch[ind].array = lut_preprocess_array(batch[ind].array, max_pix)
                        keep.append(batch[ind])
                
                self.run_batch(keep, output_volume)

    def getBatchSize(self):
        return self.batch_size

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

    def run_batch(self, batch, output_volume):
        bounding_boxes, arrays = self.toTorch(batch)
        inputs = Variable(arrays).float()

        data_list = []
        batch_list = [list(range(len(inputs)))[i:i+ int(len(batch)/10)]for i in range(0,len(inputs),int(len(batch)/10))]
        for s_batch in batch_list:
            st,end = s_batch[0], s_batch[-1]
            outputs = self.getNet()(inputs[st:end])
            data_list += self.toData(outputs, bounding_boxes[st:end])

        if hasattr(output_volume, 'tensor'):
            writes = []
            for data in data_list:
                writes.append(output_volume.blend(data))
                
            for write in writes:
                write.result()
                
        else:
            for data in data_list:
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