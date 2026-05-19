# Kisuk Lee <kisuklee@mit.edu>, 2017
# https://github.com/seung-lab/DeepEM
import numpy as np
import torch
import torch.nn as nn
from ac_segmentation.neurotorch.utils import torch_utils

class EdgeSampler(object):
    def __init__(self, edges):
        self.edges = list(edges)

    def generate_edges(self):
        return list(self.edges)

    def generate_true_aff(self, obj, edge):
        o1, o2 = torch_utils.get_pair(obj, edge)
        ret = (((o1 == o2) + (o1 != 0) + (o2 != 0)) == 3)
        return ret.type(obj.type())

    def generate_mask_aff(self, mask, edge):
        m1, m2 = torch_utils.get_pair(mask, edge)
        return (m1 * m2).type(mask.type())


class EdgeCRF(nn.Module):
    def __init__(self, criterion, size_average=False, class_balancing=False):
        super(EdgeCRF, self).__init__()
        self.criterion = criterion
        self.size_average = size_average
        self.balancing = class_balancing

    def forward(self, preds, targets): 
        assert len(preds) == len(targets) 
        loss = 0
        for pred, target in zip(preds, targets):
            l = self.criterion(pred, target)
            loss += l

        return loss

    def class_balancing(self, target, mask):
        if not self.balancing:
            return mask
        dtype = mask.type()
        m_int = mask * torch.eq(target, 1).type(dtype)
        m_ext = mask * torch.eq(target, 0).type(dtype)
        n_int = m_int.sum().item()
        n_ext = m_ext.sum().item()
        if n_int > 0 and n_ext > 0:
            m_int *= n_ext/(n_int + n_ext)
            m_ext *= n_int/(n_int + n_ext)
        return (m_int + m_ext).type(dtype)


class AffinityLoss(nn.Module):
    def __init__(self, edges, criterion, size_average=False,
                 class_balancing=False):
        super(AffinityLoss, self).__init__()
        self.sampler = EdgeSampler(edges)
        self.decoder = AffinityLoss.Decoder(edges)
        self.criterion = EdgeCRF(
            criterion,
            size_average=size_average,
            class_balancing=class_balancing
        )

    def forward(self, preds, label):
        pred_affs = list()
        true_affs = list()
        edges = self.sampler.generate_edges()
        for i, edge in enumerate(edges):
            try:
                pred_affs.append(self.decoder(preds, i))
                true_affs.append(self.sampler.generate_true_aff(label, edge))
            except:
                raise
        return self.criterion(pred_affs, true_affs)

    class Decoder(nn.Module):
        def __init__(self, edges):
            super(AffinityLoss.Decoder, self).__init__()
            assert len(edges) > 0
            self.edges = list(edges)

        def forward(self, x, i):
            num_channels = x.size(-4)
            assert num_channels == len(self.edges)
            assert i < num_channels and i >= 0
            edge = self.edges[i]
            return torch_utils.get_pair_first(x[...,[i],:,:,:], edge)
