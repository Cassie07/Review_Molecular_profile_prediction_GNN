import os
from PIL import Image
import time
import copy
import json
from math import ceil, sqrt
import random
import pandas as pd
import numpy as np
from random import randint

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torchvision.models.resnet import model_urls


from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch_geometric.data as td
import torch_geometric.transforms as T
from torch_geometric.data import DataListLoader
from torch_geometric.nn import SAGEConv, GINConv, DenseGCNConv, GCNConv, GATConv, EdgeConv
from torch_geometric.utils import dense_to_sparse, dropout_adj,true_positive, true_negative, false_positive, false_negative, precision, recall, f1_score
from torch_geometric.nn import knn_graph, dense_diff_pool, global_add_pool, global_mean_pool,DataParallel
from torch_geometric.nn import JumpingKnowledge as jp
from torch_geometric.utils import to_dense_batch


from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import metrics

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.nhid = 256
        self.feature = 64

        nn1 = Sequential(Linear(self.feature, self.nhid))#, ReLU(), Linear(self.nhid, self.nhid))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(self.nhid)

        nn2 = Sequential(Linear(self.nhid, self.nhid))#, ReLU(), Linear(self.nhid, self.nhid))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(self.nhid)

        nn3 = Sequential(Linear(self.nhid, self.nhid))#, ReLU(), Linear(self.nhid, self.nhid))
        self.conv3 = GINConv(nn3)

        self.jpn = jp('max') # 'max'
        #self.jpn = jp('lstm',self.nhid,3)


        self.fc1 = Linear(self.nhid,self.nhid//2)
        self.fc2 = Linear(self.nhid//2, self.nhid//4)
        self.fc3 = Linear(self.nhid//4, 2)

    def forward(self, data):

        h = []
        x, edge_index, batch = data.x, data.edge_index, data.batch


        x = F.relu(self.conv1(x, edge_index))
        h1 = x
        x = self.bn1(x)
        h.append(x)

        x = F.relu(self.conv2(x, edge_index))
        h2 = x
        x = self.bn2(x)
        h.append(x)

        x = F.relu(self.conv3(x, edge_index))
        h3 = x

        h.append(x)
        #print('x3')
        #x = self.jpn([h1,h2,h3])
        x = self.jpn(h)#, h4])

        #attn_x = copy.deepcopy(x)
        #attn_x = global_add_pool(attn_x, batch, 10)
        select_index = global_sort_pool(x, batch, 20)
        x = global_add_pool(x, batch)


        x = self.fc1(x)
        x = F.relu(F.dropout(x, p=0.5, training=self.training))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
#         return x, select_index
        return F.log_softmax(x, dim=-1), select_index


def global_sort_pool(x, batch, k):
    r"""The global pooling operator from the `"An End-to-End Deep Learning
    Architecture for Graph Classification"
    <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_ paper,
    where node features are sorted in descending order based on their last
    feature channel. The first :math:`k` nodes form the output of the layer.
    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        k (int): The number of nodes to hold for each graph.
    :rtype: :class:`Tensor`
    """
    fill_value = x.min().item() - 1
    batch_x, _ = to_dense_batch(x, batch, fill_value)
#     print(batch_x)
    B, N, D = batch_x.size()

    copy_all = []#copy.deepcopy(batch_x).tolist()
    for i in batch_x:
        copy_all.append(i.tolist())
    _ , perm = batch_x[:, :, -1].sort(dim=-1, descending=True)
    arange = torch.arange(B, dtype=torch.long, device=perm.device) * N
    perm = perm + arange.view(-1, 1)

    batch_x = batch_x.view(B * N, D)
    batch_x = batch_x[perm]
    batch_x = batch_x.view(B, N, D)


    if N >= k:
        batch_x = batch_x[:, :k].contiguous()
        copy_select = batch_x.tolist()
        select_index = []
        for ori_graph, k_graph in zip(copy_all, copy_select):
            node_index = []
            for node in k_graph:
                node_index.append(ori_graph.index(node))
            select_index.append(node_index)
    else:
        expand_batch_x = batch_x.new_full((B, k - N, D), fill_value)
        batch_x = torch.cat([batch_x, expand_batch_x], dim=1)

    return torch.Tensor(select_index).cuda()
