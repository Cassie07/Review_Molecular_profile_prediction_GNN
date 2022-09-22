#lustering
import numpy as np
from itertools import combinations
import bisect
from sklearn.cluster import KMeans
import pandas as pd
import json
from math import ceil, sqrt
from scipy import stats
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


thred = 85 # patch distance threshold (512 * thred)
gene = 'tp53'
label_path = './top20_coad_cptac.csv'
version = 5
pca_or_not = 'False'
feature_dimension = 512
backbone = 'resnet18'
if pca_or_not == 'True':
    feature_path = './Journal_cptac_coad_feature{}_{}_pca_v{}.json'.format(feature_dimension, backbone, version)#feature_resnet18_256_1000.json'
    feature_name_path = './Journal_cptac_coad_feature{}_{}_pca_name_v{}.json'.format(feature_dimension, backbone, version)
else:
    feature_path = './all_Journal_cptac_coad_feature{}_{}_v{}.json'.format(feature_dimension, backbone, version)#feature_resnet18_256_1000.json'
    feature_name_path = './all_Journal_cptac_coad_feature{}_{}_name_v{}.json'.format(feature_dimension, backbone, version)

# load data
dataset = pd.read_csv(label_path,delimiter=",")
df = pd.DataFrame(dataset)
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
print(false_label_slide)

path = feature_path
with open(path) as f:
    dic_slide_feature_ori = json.load(f)

# load slide:[patch]
path = feature_name_path
with open(path) as f:
    dic_slide_patch_ori = json.load(f)

#print('selected slide is {}'.format(len(dic_slide_patch.keys())))

dic_slide_feature = {}
for i,j in dic_slide_feature_ori.items():
    if i[0:7] not in false_label_slide:
        dic_slide_feature[i] = j

dic_slide_patch = {}
for i,j in dic_slide_patch_ori.items():
    if i[0:7] not in false_label_slide:
        dic_slide_patch[i] = j

        
# distance calculation: Euclidian distance
def customized_euclidian_distance(list1, list2):
    sum = 0
    for (i,j) in zip(list1, list2):
        sum += (i-j)**2
    d = sqrt(sum)
    return d

# distance calculation: Manhattan distance
def customized_manhattan_distance(list1, list2):
    sum = 0
    for (i,j) in zip(list1, list2):
        sum += abs(i-j)
    return sum

# get coordinates from patch names
def get_coordinate(name):
    #print(name)

    # if cptac
    name = name.split('/')[7]

    sp = name.split('.')[0]
    
    # if generate for READ, open #
    #sp = sp.split('/')[6]
    sp = sp.split('_')
    #print(sp)
    x = int(sp[1]) # if only patch name, use sp[1],sp[2]
    y = int(sp[2])
    return x, y




slide_patch_adj = {}

init = 1000
count = 0
for slide,patch in dic_slide_patch.items():
    if len(patch) < init:
        N = len(patch)
    else:
        N = init

    adj = np.zeros((N,N))
    NN_dic = {}
#    print(patch[0])
    for index_r,patch_r in enumerate(patch, 0):
        x_i, y_i = get_coordinate(patch_r)
        NN_dic[patch_r] = []
        for index_c, patch_c in enumerate(patch,0):
            x_j, y_j = get_coordinate(patch_c)
            d = sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
            if d < 512 * thred:
                adj[index_r][index_c] = 1
                NN_dic[patch_r].append(patch_c)
                #print((index_r, index_c)

    # update matrix
    adj_info = {}
    for i in range(len(patch)):
        adj_info[patch[i]] = []
        for j in range(len(patch)):
            adj_info[patch[i]].append(patch[j])
             
    slide_patch_adj[slide] = NN_dic#adj_info
    print('finish {}'.format(count))
    count += 1


with open('./all_Journal_cptac_coad_feature{}_{}_adj_{}_v{}.json'.format(feature_dimension, backbone, thred, version), 'w') as outfile:
    json.dump(slide_patch_adj, outfile)
