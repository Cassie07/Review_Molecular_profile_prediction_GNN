import os
import openslide
import numpy as np

from openslide.deepzoom import DeepZoomGenerator
import numpy as np
from PIL import Image, ImageDraw
import torchvision
import pandas as pd
import torch

import AllSlide
import foreground_extract_mk
import cv2

#use_gpu = torch.cuda.is_available()
#if use_gpu:
#    print("Using CUDA")
#    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
##########################################################################
# Parameter setting
folder = 'wsi_folder'
image_path='./{}'.format(folder)
result_path='./nonormlize_{}'.format(folder)
if not os.path.exists(result_path):
    os.makedirs(result_path)

step = 512
patch_size = 512
file_type = 'svs'
threshold = 0.5

##########################################################################
# Data preprocessing

# split training, validation and testing
# match label for each slide
# only consider kras in this code
# slide-level
file_list=[f for f in os.listdir(image_path)]# slide names list
print(file_list)



# summary its saving path and label
flag=0
train_tile_label=[]
train_tile_name=[]
tile_name=[]
tile_label=[]
count = 0
for s in file_list:
    slide_name=s[0:23]
    search_name=s[0:12]
    sp=image_path+'/'+s
    im_slide = AllSlide.AllSlide(sp)
    W,H=im_slide.level_dimensions[0]
    if file_type in ['kfb','svs']:
        msk = im_slide.get_foreground_mask(im_slide.level_dimensions[-1])
        msk = (np.array(msk) > 0) * 1.0
        th, tw = msk.shape[:2]
        coord_index = []
        for y in range(0, H, step):
            for x in range(0, W, step):
                ratio = th / H
                ty = int(y * ratio)
                tx = int(x * ratio)
                ps = int(patch_size * ratio)
                dy = min(ty + ps, th - 1)
                dx = min(tx + ps, tw - 1)
                if msk[ty:dy, tx:dx].mean() > 0.5:
                    coord_index.append((x, y))
    print('# of patch in this slide : ' + str(len(coord_index)))
    for (i,ele) in enumerate(coord_index,0):
        patch = im_slide.read_region(ele,0,(patch_size,patch_size)).convert('RGB')
        gray=patch.convert('L')
        bw = gray.point(lambda x: 0 if x<200 else 1, 'F')
        arr = np.array(np.asarray(bw))
        avgBkg = np.average(bw)
        if avgBkg<=threshold:
            #print(slide_name + '_' + str(i) + ".tif" + ':' + ' ' + str(avgBkg))
            #cv2.imwrite(result_path+'/'+slide_name + '_' + str(i) + ".tif",img)
            patch.save(result_path+'/'+slide_name + '_' + str(ele[0]) + '_' + str(ele[1]) + ".png")
     #       if flag==0:
     #           tile_label.append(dic_label[search_name])
            tile_name.append(result_path+'/'+slide_name + '_' + str(ele[0]) + '_' + str(ele[1]) + ".png")
    print('# of saved patches(foreground patch) : ' + str(len(tile_name)))
    #print('# of positive splitted patches : ' + str(sum(tile_label)))
    count += 1
    print('# of finished slides is: {}'.format(count))
    print()

print('======================================TOTAL SUMMARY================================================')
print('# of patches is : ' + str(len(tile_name)))
#print('# of positive patches is : ' + str(sum(tile_label)))
#print('probability of postive patches : ' + str(sum(tile_label)/len(tile_name)))

# 'train_tile_path_name.txt' : tile path(tile name contains its coordinate in the whole slide image)
with open('./split_patch_name.txt', 'w') as f:
    for item in tile_name:
        f.write("%s\n" % item)

print()
print('finish written')
