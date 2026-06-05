import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.cuda
import numpy as np
import os
import logging

class Trainer(object):
    """
    Trains a PyTorch neural network with a given input and label dataset
    """
    def __init__(self, net, inputs_volume, labels_volume, augmentation, checkpoint_dir, checkpoint_period=5000, 
                 logger_dir=None, checkpoint=None, optimizer=None, criterion=None, 
                 max_epochs=10, gpu_device=None, validation_split=0.2):
        """
        Sets up the parameters for training

        :param net: A PyTorch neural network
        :param inputs_volume: A list containing training and validation inputs
        :param labels_volume: A list containing training and validation corresponding labels
        :param checkpoint_dir: The directory to save checkpoints
        :param checkpoint_period: The number of iterations between checkpoints
        """
        self.max_epochs = max_epochs

        self.device = torch.device("cuda:{}".format(gpu_device)
                                   if gpu_device is not None
                                   else "cpu")

        self.net = net.to(self.device)

        if checkpoint is not None:
            self.net.load_state_dict(torch.load(checkpoint))

        if optimizer is None:
            self.optimizer = optim.Adam(self.net.parameters())
        else:
            self.optimizer = optimizer

        if criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = criterion

        if gpu_device is not None:
            self.gpu_device = gpu_device
            self.useGpu = True

        self.inputs_volume = inputs_volume
        self.labels_volume = labels_volume 
        
        if not os.path.isdir(checkpoint_dir):
            raise IOError("{} is not a valid directory".format(checkpoint_dir))
        
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_period = checkpoint_period
        self.max_accuracy = 0
        
        if logger_dir is not None and not os.path.isdir(logger_dir):
            raise IOError("{} is not a valid directory".format(logger_dir))
            
        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(logging.INFO)    
        
        if logger_dir:
            file_handler = logging.FileHandler(os.path.join(logger_dir,
                                                            "training.log"))
            self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        self.logger.addHandler(console_handler) 
        
        self.augmentation = augmentation
                
    def toTorch(self, arr):
        torch_arr = arr.astype(np.float)
        torch_arr = torch_arr.reshape(1, *torch_arr.shape)
        return torch_arr    
    
    def run_epoch(self, sample_batch):
        """
        Runs an epoch and saves the checkpoint if there have been enough iterations

        :param sample_batch: A dictionary containing inputs and labels with the keys 
        "input" and "label", respectively
        """
                
        inputs = Variable(sample_batch[0]).float()
        labels = Variable(sample_batch[1]).float()

        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.net(inputs)

        loss = self.criterion(torch.cat(outputs), labels)
        accuracy = torch.sum((torch.cat(outputs) > 0) & (labels > 0)).float()
        accuracy /= torch.sum((torch.cat(outputs) > 0) | (labels > 0)).float()
        loss_hist = loss.cpu().item()
        loss.backward()
        self.optimizer.step()
        
        return loss_hist, accuracy.cpu().item()
    
    def evaluate(self, batch):       
        
        with torch.no_grad():
            inputs = Variable(batch[0]).float()
            labels = Variable(batch[1]).float()

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.net(inputs)

            loss = self.criterion(torch.cat(outputs), labels)
            accuracy = torch.sum((torch.cat(outputs) > 0) & (labels > 0)).float()
            accuracy /= torch.sum((torch.cat(outputs) > 0) | (labels > 0)).float()
            
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
                self.save_checkpoint("best.ckpt")
        
        return loss.cpu().item(), accuracy.cpu().item(), torch.stack(outputs).cpu().numpy()
    
    def run_training(self):
        """
        Trains the given neural network
        """
        num_epoch = 1
        num_iter = 1
        batch_size = 3
        
        train_idx = np.random.permutation(len(self.inputs_volume[0]))
        train_idx = train_idx[:(len(train_idx) - len(train_idx) % batch_size)]
        train_idx = train_idx.reshape((-1, batch_size))
        val_idx = np.random.permutation(len(self.inputs_volume[1]))

        while num_epoch <= self.max_epochs:
            np.random.shuffle(train_idx)
            for i in range(train_idx.shape[0]):
                inputs_list = []
                labels_list = []
                for idx in train_idx[i]:
                    inputs = self.inputs_volume[0][idx]
                    labels = self.labels_volume[0][idx]
                    # Apply augmentation (optional)
                    if self.augmentation:
                        inputs, labels = self.augment(inputs, labels)
                    inputs_list.append(self.toTorch(inputs)) 
                    labels_list.append(self.toTorch(labels))
                sample_batch = [np.stack(inputs_list), np.stack(labels_list)]
                
                if num_epoch > self.max_epochs:
                    break

                print("Iteration: {}".format(num_iter))
                train_loss, train_acc = self.run_epoch([torch.from_numpy(batch.astype(np.float)) for batch in sample_batch])
                if num_iter % self.checkpoint_period == 0:
                    self.save_checkpoint("iteration_{}.ckpt".format(num_iter))
                
                if num_iter % 10 == 0:
                    val_batch = [np.stack([self.toTorch(self.inputs_volume[1][idx]) for idx in val_idx[:8]]),
                                 np.stack([self.toTorch(self.labels_volume[1][idx]) for idx in val_idx[:8]])] 
                    loss, accuracy, _ = self.evaluate([torch.from_numpy(batch.astype(np.float)) for batch in val_batch])
                    self.logger.info("Iteration: {}, Epoch: {}/{}, Train loss: {:.4f}, Train acc: {:.4f}, Test loss: {:.4f}, Test acc: {:.4f}".format(num_iter, num_epoch, self.max_epochs, train_loss, train_acc*100, loss, accuracy*100))
                    
                num_iter += 1
        
            if num_epoch == self.max_epochs:
                while self.logger.handlers:
                    self.logger.handlers.pop()
            num_epoch += 1
    
    def save_checkpoint(self, checkpoint_name):
        """
        Saves a training checkpoint
        """
        checkpoint_filename = os.path.join(self.checkpoint_dir,
                                           checkpoint_name)
        torch.save(self.net.state_dict(), checkpoint_filename)
        
    def augment(self, arr1, arr2, contrast_factor=0.3, brightness_factor=0.3):
        r = np.random.randint(4) # one of 4 orientations
        arr1 = np.rot90(arr1,r,axes=(1,2))
        arr2 = np.rot90(arr2,r,axes=(1,2))
        if np.random.randint(2): # y flip
            arr1 = np.flip(arr1,1)
            arr2 = np.flip(arr2,1)
#         if np.random.randint(2): # z flip
#             arr1 = np.flip(arr1,0)
#             arr2 = np.flip(arr2,0)
        # Intensity perturbation 
        contrast = 1 + (np.random.rand() - 0.5) * contrast_factor
        brightness = (np.random.rand() - 0.5) * brightness_factor
        arr1 = arr1 * contrast
        arr1 = arr1 + brightness*255
        arr1 = np.clip(arr1, 0, 255)
        return arr1, arr2    
