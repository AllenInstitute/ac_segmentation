import torch
from torch.autograd import Variable
import numpy as np
import joblib
from joblib import Parallel, delayed
from ac_segmentation.neurotorch.datasets.dataset import Data
from ac_segmentation.preprocess import lut_preprocess_array


class TSPredictor:
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
        self.getNet().load_state_dict(torch.load(checkpoint))
        
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        
    def getBatchSize(self):
        return self.batch_size
    
    def toTorch(self, batch):
        bounding_boxes = [data.getBoundingBox() for data in batch]
        arrays = [self.toArray(data) for data in batch]
        shapes = [i.shape for i in arrays]
                  
        arrays = torch.from_numpy(np.concatenate(arrays, axis=0))
        arrays = arrays.to(self.device)

        return bounding_boxes, arrays
    
    def toArray(self, data):
        torch_data = data.getArray().astype(float)
        torch_data = torch_data.reshape(1, 1, *torch_data.shape)
        return torch_data
    
    def toData(self, tensor_list, bounding_boxes):
        tensor = torch.cat(tensor_list).data.cpu().numpy()
        batch = [Data(tensor[i][0], bounding_box)
                 for i, bounding_box in enumerate(bounding_boxes)]

        return batch

    def run(self, input_volume, output_volume, batch_size=30, max_pix = 30000, cpus = joblib.cpu_count()):

        def para_batch(batch_index):
            keep = []
            batch = [input_volume[i] for i in batch_index]   
            batch = [Data(ts[0].result(),ts[1]) for ind,ts in enumerate(batch)]
            for ind,data in enumerate(batch):
                if np.any(data.array) == True: #skip empty arrays
                    batch[ind].array = lut_preprocess_array(batch[ind].array.astype(int), max_pix)
                    keep.append(batch[ind])
            self.run_batch(keep, output_volume)
            
        self.setBatchSize(batch_size)
        with torch.no_grad():
            batch_list = [list(range(len(input_volume)))[i:i+batch_size]
                for i in range(0,
                                len(input_volume),
                                batch_size)]
            delayed_funcs = [delayed(para_batch)(b) for b in batch_list]
            parallel_pool = Parallel(cpus)
            parallel_pool(delayed_funcs)


    def run_batch(self, batch, output_volume): 
        bounding_boxes, arrays = self.toTorch(batch)
        inputs = Variable(arrays).float()
        outputs = self.getNet()(inputs)
        data_list = self.toData(outputs, bounding_boxes)

        writes = []
        allskels = []
        for data in data_list:
            writes.append(output_volume.set(data))

        for write in writes:
            write.result()
            
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
        self.getNet().load_state_dict(torch.load(checkpoint))

    def run(self, input_volume, output_volume, batch_size=30):
        self.setBatchSize(batch_size)

        with torch.no_grad():
            batch_list = [list(range(len(input_volume)))[i:i+self.getBatchSize()]
                          for i in range(0,
                                         len(input_volume),
                                         self.getBatchSize())]

            for batch_index in batch_list:
                batch = [input_volume[i] for i in batch_index]

                self.run_batch(batch, output_volume)

    def getBatchSize(self):
        return self.batch_size

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

    def run_batch(self, batch, output_volume):
        bounding_boxes, arrays = self.toTorch(batch)
        inputs = Variable(arrays).float()

        outputs = self.getNet()(inputs)

        data_list = self.toData(outputs, bounding_boxes)
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
