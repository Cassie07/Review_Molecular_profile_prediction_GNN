import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
#import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import PIL.Image
import torchvision.models as models
#from torchsummary import summary
import torchvision.transforms as transforms
import os
#import pandas as pd
from sklearn.model_selection import train_test_split
import io
import skimage
from PIL import Image
import time
import os
import copy
#from logger import Logger
from collections import OrderedDict
from torch.optim import lr_scheduler

########################################################################
# MyDataset
class MyDataset(Dataset):
    def __init__(self, x, transform):
        self.x=x
        #self.y=y
        self.transform = transform

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img = Image.open(self.x[index])
        img = img.convert('RGB')
        #if self.transform is not None:
        img = self.transform(img)
        #img.resize_((3,224,224))
        #print(img.size())
        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        #label = torch.from_numpy(np.asarray(self.y[index]))
        return img

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)

########################################################################
# parameters setting
#transform = transforms.Compose(
#    [transforms.Resize(299),transforms.ToTensor()])
transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])


num_class=2
folder = 'segmented_patch_fold_name'
path= './nonorm_coad/{}'.format(folder)



########################################################################
# load data
#label= os.listdir(path)
imagelist=[]
label=[]
l=0
name=[]
for i in os.listdir(path):
    imagelist.append(path+'/'+i)
    name.append(i)
dataset=MyDataset(imagelist, transform)
loader = DataLoader(dataset, batch_size=300, shuffle=False,num_workers=32)

with open('./{}_tumor_pred_name.txt'.format(folder),'w') as f:
    for item in name:
        f.write("%s\n" % item)

########################################################################
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
            #print(model)
PATH='./epoch26-val_acc_0.9986.pt'
state_dict = torch.load(PATH,map_location=torch.device('cpu'))
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)

##device = torch.device("cuda:0")
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model.to(device)

#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")
#    model = torch.nn.DataParallel(model)
#    model.cuda()


print("Let's use", torch.cuda.device_count(), "GPUs!")
model = torch.nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4, amsgrad=False)

#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def adjust_lr(optimizer, epoch, num_epochs, init_lr):
    lr = init_lr * (1 - epoch / num_epochs) ** 0.9
    print('learning rate in this epoch is : ' + str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
########################################################################
# Test the network
def test(model,dataset):# scheduler, num_epochs=20):
    since = time.time()

    model.train(False)
    model.eval()
    with torch.no_grad():
        result=[]
        score=[]
        for i, data in enumerate(loader, 0):
            inputs= data.to(device)
            # forward + backward + optimize
            outputs = model(inputs)
            #_, preds = torch.max(outputs, 1)
            _, preds = torch.max(outputs.data, 1)
            for it in preds:
                result.append(it.item())

            scores, _ = torch.max(outputs.data, 1)
            for it in scores:
                score.append(it.item())
        with open('./{}_tumor_predict_label.txt'.format(folder),'w') as f:
            for item in result:
                f.write("%s\n" % item)
        with open('./{}_tumor_predict_score.txt'.format(folder),'w') as f:
            for item in score:
                f.write("%s\n" % item)
        print('Finished Testing')

test(model, dataset)#exp_lr_scheduler, num_epochs=20)
