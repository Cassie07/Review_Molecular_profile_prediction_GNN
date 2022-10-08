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
import openslide
import numpy as np
import os
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
import json
from sklearn.model_selection import StratifiedKFold
from torch.optim import lr_scheduler
import random

############################################################################################
# 1. MyDataset
class MyDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img = Image.open(self.x[index])
        img = img.convert('RGB')
        #if self.transform is not None:
        img = self.transform(img)
        #img.resize_((3,299,299))
        #print(img.size())
        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        label = torch.from_numpy(np.asarray(self.y[index]))
        return img, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
      
############################################################################################
# 2. Train_model function
def train_model(model, criterion, optimizer, num_epochs, count, memo, fold_num, memo4auc, init_lr):# scheduler, num_epochs=20):
    since = time.time()

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    best_auc = 0.0
    loss_list = []
    acc_list = []
    test_loss_list = []
    test_acc_list = []
    val_pred = []
    val_label = []
    val_score = []
    val_confusion_matrix = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_corrects = 0.0

        # train
        model.train()
        adjust_lr(optimizer, epoch, num_epochs, init_lr)
        #exp_lr_scheduler.step()

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].cuda(), data[1].cuda()
            #print(inputs.size())
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            #print(outputs)
            #_, preds = torch.max(outputs, 1)
            _, preds = torch.max(outputs.data, 1)
            #print(preds)
            #print(labels.data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #num_correct = (preds == labels).sum().item()
            #correct=num_correct/batch_size
            accuracy = (labels == preds.squeeze()).float().mean()
            # print statistics
            running_loss += loss.item()
            running_corrects += accuracy.item()
            #running_corrects += torch.sum(preds == labels.data)
            num_iter= len(y_train)/batch_size
            if (i+1) % num_iter < 1:    # print each epoch
                epoch_loss = running_loss /(i+1)
                epoch_acc = running_corrects /(i+1)
                loss_list.append(running_loss /(i+1))
                acc_list.append(running_corrects /(i+1))
                running_loss = 0.0
                running_corrects=0.0
                print('this is epoch'+' : '+str(epoch))
                print('train Loss: {:.4f} train Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        ########################################################################
        # Test the network on the test data in each epoch
        # validation
        model.eval()

        epoch_score = []
        epoch_label = []
        epoch_preds = []
        running_loss = 0.0
        running_corrects=0.0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i, data in enumerate(testloader, 0):
            #print(i)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].cuda(), data[1].cuda()

            #print(inputs.size())
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            #_, preds = torch.max(outputs, 1)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            #accuracy = torch.sum(preds == labels.data)
            accuracy = (labels == preds.squeeze()).float().mean()
            # print statistics
            running_loss += loss.item()
            running_corrects += accuracy.item()
            num_iter = len(y_val)/batch_size
            for label in labels:
                epoch_label.append(label.item())
            #val_label.append(epoch_label)
            for pred in preds:
                epoch_preds.append(pred.item())

            scores,_ = torch.max(outputs.data, 1)
            #scores_copy=scores
            for score in scores:
                epoch_score.append(score.item())
            #val_score.append(epoch_score)
                
            if (i+1) % num_iter < 1:    # print every epoch
                #auc = roc_auc_score(np.array(epoch_label), np.array(epoch_score))
                epoch_loss = running_loss /(i+1)
                epoch_acc = running_corrects /(i+1)
                test_loss_list.append(running_loss /(i+1))
                test_acc_list.append(running_corrects /(i+1))
                print('val Loss: {:.4f} val Acc: {:.4f}'.format(epoch_loss, epoch_acc))
                #print('roc_auc_score: {:.4f}'.format(auc))
                running_loss = 0.0
                running_corrects=0.0
                #if auc > best_auc:
                #    print('best auc score improves from {:.4f} to {:.4f}'.format(best_auc, auc))
                #    best_auc = auc

                #if epoch_acc > best_acc:
                save_path='./models/'+'fold{:1d}'.format(count)+'/epoch{:1d}-val_acc_{:.4f}.pt'.format(epoch,epoch_acc)
                if epoch_acc > best_acc:
                    print('best validation accuracy improves from {:.4f} to {:.4f}'.format(best_acc, epoch_acc))
                    best_acc = epoch_acc
                    best_epoch = epoch
                    #best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)
                print('*******************************************************************')
            
        for index,i in enumerate(epoch_preds, 0):
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
        epoch_confusion_matrix = [tn, tp, fn, fp]
        # memorize all info in this fold
        val_label.append(epoch_label)
        val_score.append(epoch_score)
        #epoch_confusion_matrix = [tn, tp, fn, fp]
        val_confusion_matrix.append(epoch_confusion_matrix)
        print('In epoch{:1d} total number of validation set is : {:1d}'.format(epoch, len(epoch_label)))
        print('TN : {:1d}, TP : {:1d}, FN : {:1d}, FP : {:1d}' .format(tn, tp, fn, fp))
        print()


    memo['train_loss'] = loss_list
    memo['train_acc'] = acc_list
    memo['val_loss'] = test_loss_list
    memo['val_acc'] = test_acc_list

    #time_elapsed = time.time() - since
    #print('Training complete in {:.0f}m {:.0f}s'.format(
    #    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val acc in epoch : {:1d}'.format(best_epoch))
    memo4auc['val_score'] = val_score
    memo4auc['val_label'] = val_label
    memo4auc['confusion_matrix'] = val_confusion_matrix
    print('Finish this folder')
    return memo4auc, memo

    
    #return model
    
##############################################################################################
# Parameters and path setting


transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224)
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


batch_size= 64
num_epochs= 40
num_class=2
init_lr = 1e-4



def adjust_lr(optimizer, epoch, num_epochs, init_lr):
    lr = init_lr * (1 - epoch / num_epochs) ** 0.9
    print('learning rate in this epoch is : ' + str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

##############################################################################################
# 4. Data preprocessing
directory=['ADI', 'MUC', 'MUS', 'STR', 'TUM']
root='./NCT-CRC-HE-100K-NONORM'

#######################################################################
# load data
#label= os.listdir(path)
pos_list = []
pos_label= []
neg_list = []
neg_label = []
X = []
y = []
adi = []
muc = []
mus = []
strs = []
ratio = int(14317/4)


for i in directory:
    path = root + '/' + i
    #path = i
    for j in os.listdir(path):
        if not j.startswith('.'):
            if i == 'TUM':
                pos_list.append(path + '/' + j)
                pos_label.append(1)
                X.append(path + '/' + j)
                y.append(1)
            elif i =='ADI':
                adi.append(path + '/' + j)

            elif i =='MUC':
                muc.append(path + '/' + j)

            elif i =='MUS':
                mus.append(path + '/' + j)

            else:
                strs.append(path + '/' + j)

adi_index = [k for k in range(len(adi))]
adi_index = random.sample(adi_index,ratio)
adi = np.array(adi)[np.array(adi_index)].tolist()
for item in adi:
    neg_list.append(item)
    neg_label.append(0)
    X.append(item)
    y.append(0)

muc_index = [k for k in range(len(muc))]
muc_index = random.sample(muc_index,ratio)
muc = np.array(muc)[np.array(muc_index)].tolist()
for item in muc:
    neg_list.append(item)
    neg_label.append(0)
    X.append(item)
    y.append(0)

mus_index = [k for k in range(len(mus))]
mus_index = random.sample(mus_index,ratio)
mus = np.array(mus)[np.array(mus_index)].tolist()
for item in mus:
    neg_list.append(item)
    neg_label.append(0)
    X.append(item)
    y.append(0)
    
strs_index = [k for k in range(len(strs))]
strs_index = random.sample(strs_index,ratio)
strs = np.array(strs)[np.array(strs_index)].tolist()
for item in strs:
    neg_list.append(item)
    neg_label.append(0)
    X.append(item)
    y.append(0)
                
print('# of pos(tumor region): ' + str(len(pos_list)))
print('# of neg: ' + str(len(neg_list)))
#print(label)

def balance_split(X,y):
    train_pos_index = []
    train_neg_index = []
    for index,i in enumerate(y, 0):
        if i == 1:
            train_pos_index.append(index)
        else:
            train_neg_index.append(index)

    # sampling neg(balance data)
    #train_neg_index = random.sample(train_neg_index, len(train_pos_index))
    train_neg = np.array(X)[np.array(train_neg_index)].tolist()
    train_pos = np.array(X)[np.array(train_pos_index)].tolist()
    #print(len(train_pos))
    #print(len(train_neg))
    x_train = []
    y_train = []
    for i in train_pos:
        x_train.append(i)
        y_train.append(1)
    for i in train_neg:
        x_train.append(i)
        y_train.append(0)

    # shuffle
    train_indexlist=[i for i in range(len(y_train))]
    #val_indexlist=[i for i in range(len(y_test))]
    random.shuffle(train_indexlist)
    #random.shuffle(test_indexlist)
    train_random_index = np.array(train_indexlist)
    #val_random_index = np.array(val_indexlist)
    x_train = np.array(x_train)[train_random_index].tolist()
    y_train = np.array(y_train)[train_random_index].tolist()
    #x_val = np.array(X_test)[val_random_index].tolist()
    #y_val = np.array(y_test)[val_random_index].tolist()

    return x_train, y_train

##############################################################################################
# 5. k-fold(train here)
kfold = StratifiedKFold(n_splits=5, shuffle=True)#, random_state=seed)
#print(kfold)
#cvscores = []
X_train = np.array(X)
Y = np.array(y)
count = 1
memo = {}
memo4auc = {}
for tr, te in kfold.split(X_train, Y):
    fold_num = 'Fold{:1d}'.format(count)
    print('*******************************************************************')
    print(' *** ' + fold_num + ' *** ')
    x_train=X_train[tr]
    y_train=Y[tr]
    x_train, y_train = balance_split(x_train,y_train)
    x_val=X_train[te]
    y_val=Y[te]
    print('# of training patches : '+ str(len(x_train)))
    print('# of validation patches : '+ str(len(y_val)))
    print('*******************************************************************')
    ds_train = MyDataset(x_train,y_train, transform_train)
    trainloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,num_workers=32)
    ds_val = MyDataset(x_val, y_val, transform_test)
    testloader = DataLoader(ds_val, batch_size=batch_size,shuffle=True,num_workers=32)
    print('finish load data')
    ############################################################################################
    # 3. build model
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_class)

    #for name, child in model.named_children():
    #    print(name)

    # Freeze training for all layers
    for name, child in model.named_children():
        if name in ['layer4', 'avgpool','fc']:
            print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True


    device = torch.device("cuda:0")
    model.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4, amsgrad=False)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    optimizer = optim.Adam(model.parameters(), lr = init_lr, weight_decay = 1e-4, amsgrad=False)
    memo, memo4auc = train_model(model, criterion, optimizer, num_epochs, count, memo, fold_num, memo4auc, init_lr)
    count+=1

# save memo as a json file
# track logs
with open('./logs_loss_acc.json', 'w') as outfile:
    json.dump(memo, outfile)

with open('./logs_4auc.json', 'w') as outfile:
    json.dump(memo4auc, outfile)
