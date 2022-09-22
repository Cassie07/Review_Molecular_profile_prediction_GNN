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

##############################################################################################
# Customized
from functions.model_architecture import Net, global_sort_pool
from functions.utils import generate_dataset, oversampling, majority_vote

##############################################################################################



###################################################################      Must define task name here   ######################################################################


task = 'gene'  # MSI/gene/CNA/protein


############################################################################################################################################################################
patch_thershold = 16

use_knn_graph = 'False'


if task == 'gene':
  gene_list  = ['apc','tp53','kras','pik3ca','muc16','ttn','syne1','fat4','ryr2','obscn','zfhx4','flg','lrp1b','dnah5', 'csmd1','fbxw7','csmd3','dnah11','ryr1','fat3']
elif task == 'CNA':
  gene_list = ['CCSER1', 'COX4I2', 'CSMD1', 'DEFB118', 'DUSP15', 'FOXS1', 'HCK', 'ID1', 'KIF3B', 'MACROD2', 'MYLK2', 'PDRG1', 'PLAGL2', 'POFUT1', 'RBFOX1', 'REM1', 'TM9SF4', 'TPX2', 'TSPY26P', 'WWOX']
elif task =='protein':
  gene_list = ['P53', 'PI3KP110ALPHA','EGFR','STAT3_pY705','EGFR_pY1173', 'SRC_pY416','ACC_pS79', 'TFRC', 'BRAF', 'ATM', 'PTEN', 'ERALPHA', 'HER3', 'CMET_pY1235', 'HER3_pY1289', 'FASN', 'ACC1', 'BRCA2', 'NOTCH1', 'YAP_pS127', 'AMPKALPHA', 'AMPKALPHA_pT172', 'BCL2', 'AR', 'XRCC1','ARID1A','HER2','CMYC']
else:
  gene_list = ['msi']

if task == 'MSI':
    label_path = 'MSI_MSS_coad.json'
    with open(label_path) as f:
        dic_id_label = json.load(f)

    label_path = 'MSI_MSS_read.json'
    with open(label_path) as f:
        dic_id_label2 = json.load(f)
    for k,v in dic_id_label2.items():
        dic_id_label[k] = v

else:
    if task == 'protein':
        label_path = './Protein_coadread.csv'
    elif task == 'CNA':
         label_path = './top20_coadread_CNA.csv'
    else:
         label_path = './top20_coadread.csv'
    dataset = pd.read_csv(label_path,delimiter=",")
    df = pd.DataFrame(dataset)


thred = 85
pca_or_not = 'False' # use pca in data-preprocessing
pca_right_now = 'True' # use pca in this code
pca_feature_dimension = 64 # if change, change it in model!!!!
feature_dimension = 512
backbone = 'resnet18'

batch_size = 64
node_num = 1000
init_lr = 1e-3 #3
wd = 0#1e-2 # weight decay
fold_num = 10
num_epochs = 50

###################################################################      Train and Test Function   ######################################################################

def train(train_loader, epoch, model,optimizer, device, init_lr, num_epochs):
    model.train()
#    adjust_lr(optimizer, epoch, num_epochs, init_lr)
    loss_all = 0
    correct = 0
    epoch_pred = []
    epoch_label = []

    for data_list in train_loader:
        optimizer.zero_grad()

        output,_ = model(data_list)
        y = torch.cat([data.y for data in data_list]).to(output.device)
#         loss = F.cross_entropy(output, y)

        loss = F.nll_loss(output, y)
        loss.backward()
        pred = output.max(dim=1)[1]
        correct += pred.eq(y).int().sum().item()
        loss_all += loss.item()/y.size(0)
        optimizer.step()

    return loss_all, correct / len(train_loader.dataset)


def test(loader, slide_name_order, dic_slide_patch, dataset_name):
    model.eval()
    correct = 0
    loss_all = 0
    epoch_pred = []
    epoch_label = []

    epoch_score = []
    dic_slide_select_node = {}
    for data_list in loader:
        detail = [data.data_name for data in data_list]

        output, select_index_list = model(data_list)

        y = torch.cat([data.y for data in data_list]).to(output.device)
        loss = F.nll_loss(output, y)
        pred = output.max(dim=1)[1]
        correct += pred.eq(y).int().sum().item()
        loss_all += loss.item()/y.size(0)
        sm = torch.nn.Softmax()
        score = sm(output)
        for s in score:
            epoch_score.append(s[1].item())
        for p in pred:
            epoch_pred.append(p.item())
        for l in y:
            epoch_label.append(l.item())

        select_index_list = select_index_list.tolist()
        b = []
        for i in select_index_list:
            tmp = []
            for j in i:
                tmp.append(int(j))
            b.append(tmp)
        #select_index_list = [i.item() for i in select_index_list]

        for graph,index_list in enumerate(b, 0):

            slide_name = slide_name_order[detail[graph]]
            #print(slide_name)
            node_list = dic_slide_patch[slide_name]
            #print(len(node_list))
            #print(index_list)
            select_node  = [node_list[i] for i in index_list]
            dic_slide_select_node[slide_name] = select_node

    return loss_all, correct / len(loader.dataset), epoch_pred, epoch_score, epoch_label, dic_slide_select_node
    
###################################################################      Load TCGA-COAD data    ######################################################################

start = time.time()
if pca_or_not == 'False':

    feature_path1 = './feature/all_Journal_coad_feature{}_{}_v1.json'.format(feature_dimension, backbone)
    feature_name_path1 = './feature/all_Journal_coad_feature{}_{}_name_v1.json'.format(feature_dimension, backbone)
    adj_path1 = './feature/all_Journal_coad_feature{}_{}_adj_{}_v1.json'.format(feature_dimension, backbone, thred)

    feature_path2 = './feature/all_Journal_coad_feature{}_{}_v2.json'.format(feature_dimension, backbone)
    feature_name_path2 = './feature/all_Journal_coad_feature{}_{}_name_v2.json'.format(feature_dimension, backbone)
    adj_path2 = './feature/all_Journal_coad_feature{}_{}_adj_{}_v2.json'.format(feature_dimension, backbone, thred)

    feature_path3 = './feature/all_Journal_coad_feature{}_{}_v3.json'.format(feature_dimension, backbone)
    feature_name_path3 = './feature/all_Journal_coad_feature{}_{}_name_v3.json'.format(feature_dimension, backbone)
    adj_path3 = './feature/all_Journal_coad_feature{}_{}_adj_{}_v3.json'.format(feature_dimension, backbone, thred)


    feature_path4 = './feature/all_Journal_coad_feature{}_{}_v4.json'.format(feature_dimension, backbone)
    feature_name_path4 = './feature/all_Journal_coad_feature{}_{}_name_v4.json'.format(feature_dimension, backbone)
    adj_path4 = './feature/all_Journal_coad_feature{}_{}_adj_{}_v4.json'.format(feature_dimension, backbone, thred)

    feature_path5 = './feature/all_Journal_coad_feature{}_{}_v5.json'.format(feature_dimension, backbone)
    feature_name_path5 = './feature/all_Journal_coad_feature{}_{}_name_v5.json'.format(feature_dimension, backbone)
    adj_path5 = './feature/all_Journal_coad_feature{}_{}_adj_{}_v5.json'.format(feature_dimension, backbone, thred)

else:
    feature_path1 = './feature/all_Journal_coad_feature{}_{}_pca_v1.json'.format(feature_dimension, backbone)
    feature_name_path1 = './feature/all_Journal_coad_feature{}_{}_pca_name_v1.json'.format(feature_dimension, backbone)
    adj_path1 = './feature/all_Journal_coad_feature{}_{}_pca_adj_v1.json'.format(feature_dimension, backbone)

    feature_path2 = './feature/all_Journal_coad_feature{}_{}_pca_v2.json'.format(feature_dimension, backbone)
    feature_name_path2 = './feature/all_Journal_coad_feature{}_{}_pca_name_v2.json'.format(feature_dimension, backbone)
    adj_path2 = './feature/all_Journal_coad_feature{}_{}_pca_adj_v2.json'.format(feature_dimension, backbone)

    feature_path3 = './feature/all_Journal_coad_feature{}_{}_pca_v3.json'.format(feature_dimension, backbone)
    feature_name_path3 = './feature/all_Journal_coad_feature{}_{}_pca_name_v3.json'.format(feature_dimension, backbone)
    adj_path3 = './feature/all_Journal_coad_feature{}_{}_pca_adj_v3.json'.format(feature_dimension, backbone)


    feature_path4 = './feature/all_Journal_coad_feature{}_{}_pca_v4.json'.format(feature_dimension, backbone)
    feature_name_path4 = './feature/all_Journal_coad_feature{}_{}_pca_name_v4.json'.format(feature_dimension, backbone)
    adj_path4 = './feature/all_Journal_coad_feature{}_{}_pca_adj_v4.json'.format(feature_dimension, backbone)

    feature_path5 = './feature/all_Journal_coad_feature{}_{}_pca_v5.json'.format(feature_dimension, backbone)
    feature_name_path5 = './feature/all_Journal_coad_feature{}_{}_pca_name_v5.json'.format(feature_dimension, backbone)
    adj_path5 = './feature/all_Journal_coad_feature{}_{}_pca_adj_v5.json'.format(feature_dimension, backbone)

dic_name_data1, data_name, slide_name_order1, dic_slide_patch1 = generate_dataset(feature_path1, feature_name_path1, adj_path1, use_knn_graph, pca_right_now)
print('[COAD] finish load dataset1')
dic_name_data2, data_name, slide_name_order2, dic_slide_patch2 = generate_dataset(feature_path2, feature_name_path2, adj_path2, use_knn_graph, pca_right_now)
print('[COAD]finish load dataset2')
dic_name_data3, data_name, slide_name_order3, dic_slide_patch3 = generate_dataset(feature_path3, feature_name_path3, adj_path3, use_knn_graph, pca_right_now)
print('[COAD] finish load dataset3')
dic_name_data4, data_name, slide_name_order4, dic_slide_patch4 = generate_dataset(feature_path4, feature_name_path4, adj_path4, use_knn_graph, pca_right_now)
print('[COAD] finish load dataset4')
dic_name_data5, data_name, slide_name_order5, dic_slide_patch5 = generate_dataset(feature_path5, feature_name_path5, adj_path5, use_knn_graph, pca_right_now)
print('[COAD] finish load dataset5')
#slide_name_order_list = [slide_name_order1]
#dic_slide_patch_list = [dic_slide_patch1]
slide_name_order_list = [slide_name_order1, slide_name_order2, slide_name_order3, slide_name_order4, slide_name_order5]
dic_slide_patch_list = [dic_slide_patch1, dic_slide_patch2, dic_slide_patch3, dic_slide_patch4, dic_slide_patch5]


###################################################################      Load TCGA-READ data    ######################################################################

if pca_or_not == 'False':

    feature_path1 = './feature/all_Journal_read_feature{}_{}_v1.json'.format(feature_dimension, backbone)
    feature_name_path1 = './feature/all_Journal_read_feature{}_{}_name_v1.json'.format(feature_dimension, backbone)
    adj_path1 = './feature/all_Journal_read_feature{}_{}_adj_v1.json'.format(feature_dimension, backbone)

    feature_path2 = './feature/all_Journal_read_feature{}_{}_v2.json'.format(feature_dimension, backbone)
    feature_name_path2 = './feature/all_Journal_read_feature{}_{}_name_v2.json'.format(feature_dimension, backbone)
    adj_path2 = './feature/all_Journal_read_feature{}_{}_adj_v2.json'.format(feature_dimension, backbone)

    feature_path3 = './feature/all_Journal_read_feature{}_{}_v3.json'.format(feature_dimension, backbone)
    feature_name_path3 = './feature/all_Journal_read_feature{}_{}_name_v3.json'.format(feature_dimension, backbone)
    adj_path3 = './feature/all_Journal_read_feature{}_{}_adj_v3.json'.format(feature_dimension, backbone)


    feature_path4 = './feature/all_Journal_read_feature{}_{}_v4.json'.format(feature_dimension, backbone)
    feature_name_path4 = './feature/all_Journal_read_feature{}_{}_name_v4.json'.format(feature_dimension, backbone)
    adj_path4 = './feature/all_Journal_read_feature{}_{}_adj_v4.json'.format(feature_dimension, backbone)

    feature_path5 = './feature/all_Journal_read_feature{}_{}_v5.json'.format(feature_dimension, backbone)
    feature_name_path5 = './feature/all_Journal_read_feature{}_{}_name_v5.json'.format(feature_dimension, backbone)
    adj_path5 = './feature/all_Journal_read_feature{}_{}_adj_v5.json'.format(feature_dimension, backbone)

else:
    feature_path1 = './feature/all_Journal_read_feature{}_{}_pca_v1.json'.format(feature_dimension, backbone)
    feature_name_path1 = './feature/all_Journal_read_feature{}_{}_pca_name_v1.json'.format(feature_dimension, backbone)
    adj_path1 = './feature/all_Journal_read_feature{}_{}_pca_adj_v1.json'.format(feature_dimension, backbone)

    feature_path2 = './feature/all_Journal_read_feature{}_{}_pca_v2.json'.format(feature_dimension, backbone)
    feature_name_path2 = './feature/all_Journal_read_feature{}_{}_pca_name_v2.json'.format(feature_dimension, backbone)
    adj_path2 = './feature/all_Journal_read_feature{}_{}_pca_adj_v2.json'.format(feature_dimension, backbone)

    feature_path3 = './feature/all_Journal_read_feature{}_{}_pca_v3.json'.format(feature_dimension, backbone)
    feature_name_path3 = './feature/all_Journal_read_feature{}_{}_pca_name_v3.json'.format(feature_dimension, backbone)
    adj_path3 = './feature/all_Journal_read_feature{}_{}_pca_adj_v3.json'.format(feature_dimension, backbone)


    feature_path4 = './feature/all_Journal_read_feature{}_{}_pca_v4.json'.format(feature_dimension, backbone)
    feature_name_path4 = './feature/all_Journal_read_feature{}_{}_pca_name_v4.json'.format(feature_dimension, backbone)
    adj_path4 = './feature/all_Journal_read_feature{}_{}_pca_adj_v4.json'.format(feature_dimension, backbone)

    feature_path5 = './feature/all_Journal_read_feature{}_{}_pca_v5.json'.format(feature_dimension, backbone)
    feature_name_path5 = './feature/all_Journal_read_feature{}_{}_pca_name_v5.json'.format(feature_dimension, backbone)
    adj_path5 = './feature/all_Journal_read_feature{}_{}_pca_adj_v5.json'.format(feature_dimension, backbone)

read_dic_name_data1, read_data_name, read_slide_name_order1, read_dic_slide_patch1 = generate_dataset(feature_path1, feature_name_path1, adj_path1, use_knn_graph, pca_right_now)
print('[READ] finish load dataset1')
read_dic_name_data2, read_data_name, read_slide_name_order2, read_dic_slide_patch2 = generate_dataset(feature_path2, feature_name_path2, adj_path2, use_knn_graph, pca_right_now)
print('[READ] finish load dataset2')
read_dic_name_data3, read_data_name, read_slide_name_order3, read_dic_slide_patch3 = generate_dataset(feature_path3, feature_name_path3, adj_path3, use_knn_graph, pca_right_now)
print('[READ] finish load dataset3')
read_dic_name_data4, read_data_name, read_slide_name_order4, read_dic_slide_patch4 = generate_dataset(feature_path4, feature_name_path4, adj_path4, use_knn_graph, pca_right_now)
print('[READ] finish load dataset4')
read_dic_name_data5, read_data_name, read_slide_name_order5, read_dic_slide_patch5 = generate_dataset(feature_path5, feature_name_path5, adj_path5, use_knn_graph, pca_right_now)
print('[READ] finish load dataset5')
#slide_name_order_list = [slide_name_order1]
#dic_slide_patch_list = [dic_slide_patch1]
read_slide_name_order_list = [read_slide_name_order1, read_slide_name_order2, read_slide_name_order3, read_slide_name_order4, read_slide_name_order5]
read_dic_slide_patch_list = [read_dic_slide_patch1, read_dic_slide_patch2, read_dic_slide_patch3, read_dic_slide_patch4, read_dic_slide_patch5]




###################################################################      Training start here    ######################################################################

for gene in gene_list:

    # load label
    if task != 'MSI':
        dic_id_label = {}
        name = df['id'].values.tolist()
        label = df[gene].values.tolist()
        false_label_slide = []
        for index,n in enumerate(name,0):
            if label[index] == 2:
                print(label[index])
                false_label_slide.append(n)
                continue
            else:
                dic_id_label[n]=label[index]
    #print(false_label_slide)

    # ADD label
    def add_label(dic_name_data, dic_id_label):
        pos_name_data = []
        neg_name_data = []
        name_neg = []
        name_pos = []

        for key,data in dic_name_data.items():
            id = key[0:12]
            if id in dic_id_label.keys():

                label = dic_id_label[id]
                flag = label
                y = torch.from_numpy(np.array([label]))
                print(flag)
                if flag == 1:
                    data.y = y
                    name_pos.append(key)
                    dic_name_data[key] = data
                else:
                    data.y = y
                    name_neg.append(key)
                    dic_name_data[key] = data
        return name_neg, name_pos, dic_name_data

    name_neg, name_pos, dic_name_data1 = add_label(dic_name_data1, dic_id_label)
    #print(name_neg)
    #print(name_pos)
    name_neg, name_pos, dic_name_data2 = add_label(dic_name_data2, dic_id_label)
    name_neg, name_pos, dic_name_data3 = add_label(dic_name_data3, dic_id_label)
    name_neg, name_pos, dic_name_data4 = add_label(dic_name_data4, dic_id_label)
    name_neg, name_pos, dic_name_data5 = add_label(dic_name_data5, dic_id_label)
    #dic_name_data_list = [dic_name_data1]
    dic_name_data_list = [dic_name_data1, dic_name_data2, dic_name_data3, dic_name_data4, dic_name_data5]

    read_name_neg, read_name_pos, read_dic_name_data1 = add_label(read_dic_name_data1, dic_id_label)
    read_name_neg, read_name_pos, read_dic_name_data2 = add_label(read_dic_name_data2, dic_id_label)
    read_name_neg, read_name_pos, read_dic_name_data3 = add_label(read_dic_name_data3, dic_id_label)
    read_name_neg, read_name_pos, read_dic_name_data4 = add_label(read_dic_name_data4, dic_id_label)
    read_name_neg, read_name_pos, read_dic_name_data5 = add_label(read_dic_name_data5, dic_id_label)

    #read_dic_name_data_list = [dic_name_data1]
    read_dic_name_data_list = [read_dic_name_data1, read_dic_name_data2, read_dic_name_data3, read_dic_name_data4, read_dic_name_data5]

    dic_multi_model_pred = {}

    for i in read_name_pos:
        dic_multi_model_pred[i] = []

    for i in read_name_neg:
        dic_multi_model_pred[i] = []


    dic_multi_model_score = {}

    for i in read_name_pos:
        dic_multi_model_score[i] = []

    for i in read_name_neg:
        dic_multi_model_score[i] = []

    dic_multi_model_attn = {}

    for i in read_name_pos:
       dic_multi_model_attn[i] = []

    for i in read_name_neg:
       dic_multi_model_attn[i] = []

    # k = 1
    # result_k = {}
    # index_pos = np.array([i for i in range(len(name_pos))])
    # kf = KFold(n_splits= fold_num, random_state = 42, shuffle=True)
    #
    # index_neg = np.array([i for i in range(len(name_neg))])
    # #kf_neg = KFold(n_splits=fold_num, random_state=None, shuffle=True)
    #
    # for pos,neg in zip(kf.split(index_pos),kf.split(index_neg)):
    m = 1
    result_m = {}


    # =====================================================================================================================
    # balance
    if len(name_pos) > len(name_neg):
        train_pos, train_neg = oversampling(name_pos, name_neg)
    else:
        train_neg, train_pos = oversampling(name_neg, name_pos)
    # =====================================================================================================================


    for dic_name_data, slide_name_order, dic_slide_patch in zip(dic_name_data_list, slide_name_order_list, dic_slide_patch_list):
        print()
        print('【{} : model {} 】'.format(gene, m))
        #result_m = {}
        # train_pos, test_pos = pos[0], pos[1]
        # train_neg, test_neg = neg[0], neg[1]

        train_dataset = []
        train_name = []
        test_name = []
        test_dataset = []

        for i in train_pos:
            #print(dic_name_data[i])
            train_dataset.append(dic_name_data[i])
            train_name.append(i)

        for i in train_neg:
            train_dataset.append(dic_name_data[i])
            train_name.append(i)

        random.shuffle(train_dataset)

        #test_dataset = list(read_dic_name_data_list[m - 1].values())
        #print(test_dataset[0])
        #print(test_dataset[0].y)

        if task == 'protein' or task == 'MSI':
            test_name = read_name_neg + read_name_pos
            test_dataset = []
            for k, v in read_dic_name_data_list[m - 1].items():
                if k in test_name:
                    test_dataset.append(v)
        else:
            test_name = list(read_dic_name_data_list[m - 1].keys())
            test_dataset = list(read_dic_name_data_list[m - 1].values())



        #print('fold {}'.format(k))
        print('# of train: {} | # of test{}'.format(len(train_dataset), len(test_dataset)))
        print('pos/neg in train : {}'.format(len(train_pos)/len(train_neg)))
        #print('# of sampled is {}(not use)'.format(len(not_use_list)))
        print('pos/neg in test : {}'.format(len(read_name_pos)/len(read_name_neg)))
        print()
        #test_loader = graphDataLoader(test_dataset, batch_size = 64, shuffle = True)#, num_workers=32)
        train_loader = DataListLoader(train_dataset, batch_size = batch_size, shuffle = True)#, num_worker=32)
        test_loader = DataListLoader(test_dataset, batch_size = batch_size, shuffle = False)
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = Net()#.to(device)
        for layer in model.children():
            layer.reset_parameters()
        #model = GCNSAG_h()#.to(args.device)
        #if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay = wd)
        #optimizer = torch.optim.SGD(model.parameters(), lr = init_lr, momentum=0.9)#, weight_decay = 1e-4)
        num_classes = 2

        best_test_acc = test_acc = best_test_auc = best_auc =  0
        best_test_epoch = 0
        save_pos_acc = 0
        save_neg_acc = 0
        best_auc_epoch = 0
        save_pos_acc_inauc = 0
        save_neg_acc_inauc = 0

        train_accuracy = []
        train_losses = []


        test_accuracy  = []
        test_losses = []



        for epoch in range(num_epochs):
            train_loss, train_acc= train(train_loader, epoch, model, optimizer, device, init_lr, num_epochs, aug_in_loader)
            test_loss, test_acc, pred_label, pred_score, epoch_label, epoch_slide_select_patch = test(test_loader, read_slide_name_order_list[m-1], read_dic_slide_patch_list[m-1], dataset_name = 'testing')

            y = epoch_label
            y_score = pred_score
            fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
            test_auc = metrics.auc(fpr, tpr)
            print('Epoch: {:03d} :'.format(epoch+1))
            print('Train Loss: {:.7f} | Train acc: {:.7f} |Test Loss: {:.7f} | Test Acc : {} | Test Auc: {:.7f}'.format(train_loss, train_acc, test_loss, test_acc, test_auc))
            print()

            if test_auc > best_test_auc:
                best_test_acc = test_acc
                best_test_auc = test_auc
                prediction = pred_label
                score = pred_score
                dic_slide_select_patch = epoch_slide_select_patch

        tn = 0
        fn = 0
        tp = 0
        fp = 0

        if len(prediction) == 0:
            prediction = pred_label
            score = pred_score
            dic_slide_select_patch = epoch_slide_select_patch


        for index,i in enumerate(prediction, 0):
            if epoch_label[index] == i:
                if i == 0: # TN: true negative
                    tn += 1
                else:           # TP: true positive
                    tp += 1
            else:
                if i == 0: # FN: false negative
                    fn += 1
                else:           # FP: false positive
                    fp += 1
        acc = (tp + tn)/(tp + tn +fp + fn)
        y = np.array(epoch_label)
        y_score = np.array(score)
        fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)

        auc = metrics.auc(fpr, tpr)


        for names, labels in zip(test_name, prediction):
            dic_multi_model_pred[names].append(labels)

        for names, scores in zip(test_name, score):
            dic_multi_model_score[names].append(scores)

        for names in test_name:
            if names not in dic_slide_select_patch.keys():
                print(names)
                continue
            else:
                dic_multi_model_attn[names].append(list(dic_slide_select_patch[names]))

        result_m[m] = [best_test_acc, auc,tp, tn, fp, fn] # result_m = {'1':[...], '2':[...], ...}
        m += 1
        # result_k[k] = result_m #{'1':{'1':[acc, auc, ...], '2':[acc,auc,...],...], '2':[]}}
        # k += 1
    # ensemble five prediction results
    ground_truth = []
    y_pred = []
    tn = 0
    fn = 0
    tp = 0
    fp = 0
    y_score = []
    name_list = []
    for name, pred_list in dic_multi_model_pred.items():
        ids = name[0:12]
        label = dic_id_label[ids]
        ground_truth.append(label)
#        s = dic_multi_model_score[name]
#        flag,ave = majority_vote2(pred_list, s,3)
        flag = majority_vote(pred_list, 3)
#        flag = pred_list[0]
        y_pred.append(flag)
        s = dic_multi_model_score[name]
        ave = sum(s)/len(s)
        y_score.append(ave)
        name_list.append(name)

    correct_pos = []
    correct_neg = []
    for index,i in enumerate(y_pred, 0):
        if ground_truth[index] == i:
            if i == 0: # TN: true negative
                tn += 1
                correct_neg.append(name_list[index])
            else:           # TP: true positive
                tp += 1
                correct_pos.append(name_list[index])
        else:
            if i == 0: # FN: false negative
                fn += 1
            else:           # FP: false positive
                fp += 1

    acc = (tp + tn)/(tp + tn +fp + fn)
    y = np.array(ground_truth)
    scores = np.array(y_score)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    #print(tpr)
    #print(fpr)
    auc = metrics.auc(fpr, tpr)
    #a = tp+fn
    #if a == 0:
    #    recall = 0.5
    #else:
    #    recall = tp / (tp + fn)
    #b = tp+fp

    #if b == 0:
    #    precision = 0.5
    #else:
    #    precision = tp / (tp + fp)
    #specificity = tn/(tn + fp)
    #f1_score = 2 * (precision * recall / (precision + recall))
    #print('【Current Gene is {}】The ensemble result is : acc is {}, auc is {}, recall is {}, precision is {}, specificity is {}, f1_score is {}'.format(gene, acc, auc, recall, precision, specificity, f1_score))
    print('【Current Gene is {}】The ensemble result is : acc is {}, auc is {}'.format(gene, acc, auc))
    print('tp : {}, tn: {}, fp: {}, fn:{}'.format(tp, tn, fp, fn))

    gene_ensemble_result[gene] = [acc, auc, tp, tn, fp, fn]#, recall, precision, specificity, f1_score, tp, tn, fp, fn]
    log = {'ground_truth': ground_truth, 'pred_label': y_pred, 'score': y_score}
    # attn[gene] = dic_multi_model_attn
    result_gene[gene] = result_m # {'gene' : {'1' : {'1':[acc,auc,....], '2':[...]}, '2':{...}}}

    #with open('./{}_pca{}_no_drop_10fold_lr1e-3_log.json'.format(gene,pca_feature_dimension), 'w') as outfile:
    #   json.dump(log, outfile)

    #with open('./{}_pca{}_no_drop_10fold_lr1e-3_top20_selected_patch.json'.format(gene,pca_feature_dimension), 'w') as outfile:
    #    json.dump(dic_multi_model_attn, outfile)

stop = time.time()

for gene, result_m in result_gene.items():
    print('【{}】'.format(gene))
    sum_acc = 0
    sum_auc = 0
    for num_model, r in result_m.items():
        print('(model {}) : acc is {}, auc is {}, tp : {}, tn: {}, fp: {}, fn:{}'.format(num_model, r[0],r[1],r[2],r[3],r[4],r[5]))
        sum_acc += r[0]
        sum_auc += r[1]
    print('【average acc for 5 model is {}, average auc for 5 model is {}】'.format(sum_acc/5, sum_auc/5))

for gene in gene_list:
    r = gene_ensemble_result[gene]
    #print('【 {} 】 : acc is {}, auc is {}, recall is {}, precision is {}, specificity is {}, f1_score is {}'.format(gene, r[0],r[1],r[2],r[3],r[4],r[5]))
    print('【 {} 】 : acc is {}, auc is {}').format(gene, r[0],r[1])
    print('tp : {}, tn: {}, fp: {}, fn:{}'.format(r[2],r[3],r[4],r[5]))
    print()

hr = (stop - start)/3600
print('training time : {}'.format(hr))
