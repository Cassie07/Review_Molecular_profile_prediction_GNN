import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
#import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import PIL.Image
import torchvision.models as models
#from torchsummary import summary
import torchvision.transforms as transforms
#import pandas as pd
from sklearn.model_selection import train_test_split
import io
import skimage
from PIL import Image
import time
import copy
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from torchvision.models.resnet import model_urls
import pandas as pd

import numpy as np
from torch.nn import Sequential, Linear, ReLU
import torch_geometric.data as td
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader#,DataLoader
from torch_geometric.data import DataLoader as graphDataLoader
from torch_geometric.nn import DenseSAGEConv, GINConv, DenseGCNConv, GCNConv
from math import ceil, sqrt
import random
from sklearn.decomposition import PCA
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import dense_diff_pool, global_add_pool, global_mean_pool
import json


# Some threshold and pre-setting parameters
patch_select_threshold = 64 # A threshold to select patient with enough patches. We will discard patient without enough patches (e.g., num_patch < patch_select_threshold). 
pca_dimension = 64
gene = 'apc'
backbone = 'resnet18'
pca_or_not = 'False' 
feature_dimension = 512 
version = 5 # group number of patches: we have five group of patches to generate five subgraphs


############################################################################################
# Prepare dataset 
class MyDataset(Dataset):
    def __init__(self, x, transform):
        self.x = x
        #self.y = y
        self.transform = transform

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img = Image.open(self.x[index])
        #print(tuple(img.size))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        #if self.transform is not None:
        img = self.transform(img)
        #print(img.size())
        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
       # label = torch.from_numpy(np.asarray(self.y[index]))
        return img#, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)

###########################################################################################

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

##############################################################################################
# 4. Data preprocessing
# load label
label_path = './top20_coadread.csv'#top20_coadread.csv'#coad_new.csv'#summary(ffpe).csv'#coad_new.csv'#read_new.csv'#ummary(ffpe).csv'
dataset = pd.read_csv(label_path,delimiter=",")
df = pd.DataFrame(dataset)
dic_id_label = {}
name = df['id'].values.tolist()
label = df[gene].values.tolist()
# {'TCGA-cc-dddd':0, 'TCGA-cc-bbbb':1}
false_label_slide = []
for index,n in enumerate(name,0):
    if label[index] == 2:
        print(label[index])
        false_label_slide.append(n)
        continue
    else:
        dic_id_label[n]=label[index]
print()
#print(dic_id_label.keys())
image_path = './normalization_patches'#_READ'#transform2'

# load slide names
old_patch_name = [i for i in os.listdir(image_path)]
print(old_patch_name[0])
patch_name = [i for i in old_patch_name if i[0:7] not in false_label_slide]
slides = [i[0:23] for i in patch_name]
#print(list(set(slides)))

# build slide_patch dictionary
dic_slide_patch = {}
for i in patch_name:
    sn = i[0:23]
    if sn in dic_slide_patch.keys():
        dic_slide_patch[sn].append(image_path + '/' + i)
    else:
        dic_slide_patch[sn] = [image_path + '/' + i]
#print(sn)

#image_path = './transform_READ_2'#transform2'

# load slide names
#old_patch_name = [i for i in os.listdir(image_path)]
#print(patch_name[0])
#patch_name = [i for i in old_patch_name if i[0:12] not in false_label_slide]
# build slide_patch dictionary
#for i in patch_name:
#    sn = i[0:23]
#    if sn in dic_slide_patch.keys():
#        if i not in dic_slide_patch[sn]:
#            dic_slide_patch[sn].append(image_path + '/'+i)
#    else:
#        dic_slide_patch[sn] = [image_path + '/'+i]


#image_path = './transform_READ_3'#transform2'

# load slide names
#old_patch_name = [i for i in os.listdir(image_path)]
#print(patch_name[0])
#patch_name = [i for i in old_patch_name if i[0:12] not in false_label_slide]
# build slide_patch dictionary
#for i in patch_name:
#    sn = i[0:23]
#    if sn in dic_slide_patch.keys():
#        if i not in dic_slide_patch[sn]:
#            dic_slide_patch[sn].append(image_path + '/'+i)
#    else:
#        dic_slide_patch[sn] = [image_path + '/' + i]

#print(len(dic_slide_patch.keys()))
# minimum number of patch = patch_select_threshold
dic_selected_patch = {}
not_use_slide = []
for key,patches in dic_slide_patch.items():
    if len(patches) < patch_select_threshold:
        not_use_slide.append(key)
        continue
    else:
        dic_selected_patch[key] = patches
print('selected slide is {} | not used slide is {}'.format(len(dic_selected_patch.keys()), len(not_use_slide)))

patient = [i[0:7] for i in dic_selected_patch.keys()]
print(' # of patients: {}'.format(len(set(patient))))

sum_not = 0
sum_select = 0
no_label_data = []
for i in not_use_slide:
    sid = i[0:7]
    if sid not in list(dic_id_label.keys()):
        no_label_data.append(sid)
    else:
        l = dic_id_label[sid]
        sum_not += l
print('pos not used slide is {}'.format(sum_not))

for i in dic_selected_patch.keys():
    sid = i[0:7]
    if sid not in list(dic_id_label.keys()):
        no_label_data.append(sid)
    else:
        l = dic_id_label[sid]
        sum_select += l
print('pos selected slide is {}'.format(sum_select))
print('no label data is {}'.format(len(no_label_data)))

# use resnet18 as feature extractor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model = nn.DataParallel(model)
model.to(device)
#    output = model(img)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(len(dic_selected_patch.keys()))

# get coordinate
def get_coordinate(name):
    sp = name.split('.')[0]
    sp = sp.split('_')
    x = int(sp[1])
    y = int(sp[2])
    return x, y

# calculate Euclidean Distance and generate adjaceny matrix
# TCGA-5M-AAT5-01Z-00-DX1_20480_12288.png
def generate_adj(patch_list, N):
    adj = np.zeros((N,N))
    for index_r,patch_r in enumerate(patch_list, 0):
        x_i, y_i = get_coordinate(patch_r)
        for index_c, patch_c in enumerate(patch_list,0):
            x_j, y_j = get_coordinate(patch_c)
            d = sqrt( (x_i - x_j)**2 + (y_i - y_j)**2 )
            if d < 512*3:
                adj[index_r][index_c] = 1
                #print((index_r, index_c))
    return adj

# build dataset
data_list_pos = []
data_list_neg = []
feature = {}
use_patch = {}
count = 0
num = 0
for key, patch in dic_selected_patch.items():
    # X: extract feature
    #patchlist = [image_path + '/' + i for i in patch]
    
    patchlist = patch
    N = len(patchlist)
    print(N)
    if N > 1000:
        N = 1000
        num += 1
    # random sampling
    indexlist=[i for i in range(len(patchlist))]
    random.shuffle(indexlist)
    indexlist = random.sample(indexlist, N)
    random_index = np.array(indexlist)
    imagelist = np.array(patchlist)[random_index].tolist()
    use_patch[key] = np.array(patch)[random_index].tolist()
    #use_patch[key] = patch
  
    #imagelist = patchlist
    dataset= MyDataset(imagelist, transform)
    loader = DataLoader(dataset, batch_size = 128, shuffle=False,num_workers=32)
    all_output = []
    for index, data in enumerate(loader, 0):
        inputs = data.to(device)
        output = model(inputs).squeeze()
        output = output.detach().cpu().clone().numpy()
        #print(tuple(output.shape))
        if len(output) != 512: #256#4096
            for i in output:
                all_output.append(i)
        else:
            all_output.append(output)
    #for j in output:
    #        all_output.append(j)
    all_output = np.array(all_output)
    x = all_output    
    
    if pca_or_not == 'True':
        x = PCA(n_components = pca_dimension).fit_transform(all_output)
    count += 1
    print('finish {}'.format(count))
    feature[key] = x.tolist()
print()
print('# of patches larger than 1000 : {}'.format(num))

if pca_or_not == 'True':
    with open('./all_Journal_cptac_coad_feature{}_{}_pca_v{}.json'.format(patch_threshold, backbone, version), 'w') as outfile:
            json.dump(feature, outfile)

    with open('./all_Journal_cptac_coad_feature{}_{}_pca_name_v{}.json'.format(patch_threshold, backbone, version), 'w') as outfile:
            json.dump(use_patch, outfile)
else:
    with open('./all_Journal_cptac_coad_feature{}_{}_v{}.json'.format(feature_dimension, backbone, version), 'w') as outfile:#
            json.dump(feature, outfile)

    with open('./all_Journal_cptac_coad_feature{}_{}_name_v{}.json'.format(feature_dimension, backbone, version), 'w') as outfile:
            json.dump(use_patch, outfile)
print('finish')

