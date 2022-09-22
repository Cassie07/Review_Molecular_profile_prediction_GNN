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

def generate_dataset(pca_feature_dimension, feature_path, feature_name_path, slide_patch_adj_path, use_knn_graph, pca_right_now):
    # load slide:feature
    path = feature_path
    with open(path) as f:
        dic_slide_feature = json.load(f)

    # load slide:[patch]
    path = feature_name_path
    with open(path) as f:
        dic_slide_patch = json.load(f)


    path = slide_patch_adj_path
    with open(path) as f:
        dic_slide_adj_info = json.load(f)

    # build dataset according to the selection slides
    data_list_pos = []
    data_list_neg = []
    data_pos_name = []
    data_neg_name = []
    pos_name_data = {}
    neg_name_data = {}

    data_list = []
    data_name = []
    dic_name_data = {}

    feature = {}
    slide_name_order = []
    count = 0

    for key, patch in dic_slide_patch.items():
        imagelist = patch
        #print(imagelist[0])
        N = len(imagelist)

        # X: extract feature
        x = dic_slide_feature[key]
        print(len(x))
        if len(x) < pca_feature_dimension:
            continue
        if pca_right_now == 'True':
            x = np.array(x)
            x = PCA(n_components = pca_feature_dimension).fit_transform(x)

        x = torch.Tensor(x)
        #print(tuple(x.shape))

        # y: label
        id = key[0:12]
        # label = dic_id_label[id]
        # flag = label
        # y = torch.from_numpy(np.array([label]))

        # A: Euclidean Distance based Adjacency Matrix
        #adj = generate_feature_adj(copy_x, N)
        if use_knn_graph == 'True':
            patch_name_list = dic_slide_patch[key]
            coordinates = torch.Tensor([get_coordinate(i) for i in patch_name_list])
            batch = torch.LongTensor(np.zeros(len(patch_name_list)))
            edge_index = knn_graph(coordinates, k = 500, batch=batch, loop=True)

        else:
            adj_info = dic_slide_adj_info[key]
            adj = generate_adj(imagelist, N, adj_info)
            #adj = np.random.randint(2, size=(100,100))
            adj = torch.Tensor(adj)
            edge_index, edge_attr = dense_to_sparse(adj)
        slide_name_order.append(key)
        data = td.Data(edge_index = edge_index, x = x)
        data.data_name = torch.tensor([count],dtype=torch.int)

        count += 1

        print('count: {}, slide: {}'.format(count, key))
        # drop edge
        #edge_index = dropout_adj(edge_index, p=0.7)[0]

        #data = td.Data(edge_index = edge_index, x = x)
        #data.data_name = torch.tensor([count],dtype=torch.int)
        #print(data)
        # if flag == 1:
        #     data_list_pos.append(data)
        #     data_pos_name.append(key)
        #     pos_name_data[key] = data
        # else:
        #     data_list_neg.append(data)
        #     data_neg_name.append(key)
        #     neg_name_data[key] = data
        data_list.append(data)
        data_name.append(key)
        dic_name_data[key] = data

    return dic_name_data, data_name, slide_name_order, dic_slide_patch  #pos_name_data, neg_name_data, data_pos_name, data_neg_name, slide_name_order, dic_slide_patch

# calculate Euclidean Distance and generate adjaceny matrix
# TCGA-5M-AAT5-01Z-00-DX1_20480_12288.png
def generate_adj(patch_list, N, adj_info):
    adj = np.zeros((N,N))
    for index_r,patch_r in enumerate(patch_list, 0):
        neighbors = adj_info[patch_r]
        for index_c, patch_c in enumerate(patch_list,0):
            if index_r == index_c:
                adj[index_r][index_c] = 1
            if patch_c in neighbors:
                adj[index_r][index_c] = 1
    return adj

def majority_vote(list1, thred):
    sums = sum(list1)
    flag = 0
    if sums > thred or sums == thred:
        flag = 1
    else:
        flag = 0
    return flag


# oversampling
def oversampling(list1, list2): # list1 > list2
    N1 = len(list1)
    N2 = len(list2)
    d = N1 - N2
    repeat = []
    list1 = list(list1)
    list2 = list(list2)
    for i in range(d):
        r = random.choice(list2)
        repeat.append(r)
    for i in repeat:
        list2.append(i)
    random.shuffle(list2)
    return list1, list2# repeat

def random_sampling(list1, list2): # of list1 > # of list2
    N = len(list2)
    old_indexlist=[i for i in range(len(list1))]
    random.shuffle(old_indexlist)
    indexlist = random.sample(old_indexlist, N)
    #not_use_index = []
    not_list = []
    new_list1 = []
    for i in range(len(list1)):
        if i in indexlist:
            new_list1.append(list1[i])
        else:
            not_list.append(list1[i])
    return new_list1, list2


# sampling + oversampling
def both(list1, list2): # list1 > list2
    N1 = len(list1)
    N2 = len(list2)
    s = int((N1 + N2)/2)
    fake_list = [i for i in range(s)]
    list1, fake_list = random_sampling(list1,fake_list)
    fake_list, list2 = oversampling(fake_list, list2)
    return list1, list2

def majority_vote2(list1, score, thred):
    sums = sum(list1)
    flag = 0
    if sums >= thred:
        flag = 1
    else:
        flag = 0
    score_all = 0
    count = 0
    for label,s in zip(list1, score):
        if label == flag:
            count += 1
            score_all += s
    return flag,score_all/count

# get coordinate
def get_coordinate(name):
    #print(name)
    sp = name.split('.')[0]
    sp = sp.split('_')
    x = int(sp[1])
    y = int(sp[2])
    return x, y


def customized_euclidian_distance(list1, list2):
    sum = 0
    for (i,j) in zip(list1, list2):
        sum += (i-j)**2
    d = sqrt(sum)
    return d

def customized_manhattan_distance(list1, list2):
    sum = 0
    for (i,j) in zip(list1, list2):
        sum += abs(i-j)
    return sum

def generate_feature_adj(x_reduced, N):
    distance = []
    adj = np.zeros((N,N))
    for index_r,feature_r in enumerate(x_reduced, 0):
        for index_c, feature_c in enumerate(x_reduced,0):
            #d = customized_euclidian_distance(feature_r, feature_c)
            d = customized_manhattan_distance(feature_r, feature_c)
            if d < 150:
                adj[index_r][index_c] = 1
                #print((index_r, index_c))
            distance.append(d)
    return adj#, distance


def dropedge(datalist, p):
    new_datalist = []
    for data in datalist:
        edge_index,x,y,data_name = data.edge_index,data.x, data.y, data.data_name
        edge_index = dropout_adj(edge_index, p = p)[0]
        new_data = td.Data(edge_index = edge_index, x = x, y = y, data_name = data_name)
        new_datalist.append(new_data)
    return new_datalist


def adjust_lr(optimizer, epoch, num_epochs, init_lr):
    lr = init_lr * (1 - epoch / num_epochs) ** 0.9
    print('learning rate in this epoch is : ' + str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
