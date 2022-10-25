import pandas as pd
import json
import os

files = [i.split('.')[0] for i in os.listdir('./directory_includes_all_molecular_info')]
print(files)


path = './file_coad_patient_name'
with open(path) as f:
    dic_slide_patch = json.load(f)
coad_used = list(dic_slide_patch.keys())


path = './file_read_patient_name'
with open(path) as f:
    dic_slide_patch = json.load(f)
read_used = list(dic_slide_patch.keys())


label_path = './file_patient_names_with_mutation'
dataset = pd.read_csv(label_path,delimiter="\t")
df = pd.DataFrame(dataset)
dic_id_label = {}
names = df['Patient ID'].values.tolist()

print('【 COAD AND READ SUMMARY】')
print('total number of patients: ' + str(len(names)))
# types = df['TCGA PanCanAtlas Cancer Type Acronym'].values.tolist()


# count = 0
# for i in coad_used:
#     if i[0:12] in names:
#         count += 1
# print('total number of patients in COAD we have: ' + str(count))


# count = 0
# for i in read_used:
#     if i[0:12] in names:
#         count += 1
# print('total number of patients in READ we have: ' + str(count))


# load Patient names
def load_name(path):
    dataset = pd.read_csv(path,delimiter="\t")
    df = pd.DataFrame(dataset)
    names = df['Patient ID'].values.tolist()
    return names

# Combine molecular profiles from different source
def union_lists(gene_list1, gene_list2):#, gene_list3):
    c = set(gene_list1).union(set(gene_list2))
    #d = set(gene_list3).union(c)
    return list(c)

# Return patient with specified molecular profile alteration
# Print the summary of molecular profile ratio
def Summary_combine(gene, coad_used, read_used):
    apc1 = load_name('/Users/kexinding/Downloads/CNA_data/{}.tsv'.format(gene))
    apc2 = load_name('/Users/kexinding/Downloads/CNA_data_f/{}.tsv'.format(gene))
#     apc3 = load_name('/Users/kexinding/Downloads/Point_mutation_meta_data_nature/{}.tsv'.format(gene))
#     count = 0
#     for i in coad_used:
#         if i[0:12] in apc2:
#             count += 1
    apc = union_lists(apc1,apc2)#,apc3)
    print()
    print('【 {} 】'.format(gene))
    print('total number of patients of {} mutation: {}' .format(gene, len(apc)))
    print('Mutation rate of {} : {}'.format(gene, str(len(apc)/len(names))))

    count = 0
    for i in coad_used:
        if i[0:12] in apc:
            count += 1
    print('total number of slides of {} mutation we have in COAD : {}'.format(gene, count))
    print('Mutation rate of {} we have in COAD : {}'.format(gene, count/274))

    count = 0
    for i in read_used:
        if i[0:12] in apc:
            count += 1
    print('total number of slides of {} mutation we have in READ : {}'.format(gene, count))
    print('Mutation rate of {} we have in READ : {}'.format(gene, count/74))
    return apc

# Achieve patient name with specified molecar profile
# You can change the following code depends on your task: Gene/CNA/Protein/MSI
rbfox1 = Summary_combine('rbfox1',coad_used, read_used)
wwox = Summary_combine('wwox',coad_used, read_used)
CCSER1 = Summary_combine('ccser1',coad_used, read_used)
CSMD1 = Summary_combine('csmd1',coad_used, read_used)
TM9SF4 = Summary_combine('tm9sf4',coad_used, read_used)
MACROD2 = Summary_combine('MACROD2',coad_used, read_used)
HCK = Summary_combine('HCK',coad_used, read_used)
PLAGL2 = Summary_combine('PLAGL2',coad_used, read_used)
TPX2= Summary_combine('TPX2',coad_used, read_used)
POFUT1= Summary_combine('POFUT1',coad_used, read_used)
TSPY26P= Summary_combine('TSPY26P',coad_used, read_used)
FOXS1= Summary_combine('FOXS1',coad_used, read_used)
ID1 = Summary_combine('ID1',coad_used, read_used)
KIF3B = Summary_combine('KIF3B',coad_used, read_used)
REM1 = Summary_combine('REM1',coad_used, read_used)
PDRG1 = Summary_combine('PDRG1',coad_used, read_used)
COX4I2 = Summary_combine('COX4I2',coad_used, read_used)
MYLK2 = Summary_combine('MYLK2',coad_used, read_used)
DEFB118 = Summary_combine('DEFB118',coad_used, read_used)
DUSP15 = Summary_combine('DUSP15',coad_used, read_used)
# apc = Summary_combine('apc', coad_used, read_used)
# tp53 = Summary_combine('tp53', coad_used, read_used)
# ttn = Summary_combine('ttn', coad_used, read_used)
# kras = Summary_combine('kras', coad_used, read_used)
# pik3ca = Summary_combine('pik3ca', coad_used, read_used)
# muc16 = Summary_combine('muc16', coad_used, read_used)
# syne1 = Summary_combine('syne1', coad_used, read_used)
# fat4 = Summary_combine('fat4', coad_used, read_used)
# ryr2 = Summary_combine('ryr2', coad_used, read_used)
# obscn = Summary_combine('obscn', coad_used, read_used)
# zfhx4 = Summary_combine('zfhx4', coad_used, read_used)
# flg = Summary_combine('flg', coad_used, read_used)
# lrp1b = Summary_combine('lrp1b', coad_used, read_used)
# dnah5 = Summary_combine('dnah5', coad_used, read_used)
# csmd1 = Summary_combine('csmd1', coad_used, read_used)
# fbxw7 = Summary_combine('fbxw7', coad_used, read_used)
# csmd3 = Summary_combine('csmd3', coad_used, read_used)
# dnah11 = Summary_combine('dnah11', coad_used, read_used)
# ryr1 = Summary_combine('ryr1', coad_used, read_used)
# fat3 = Summary_combine('fat3', coad_used, read_used)


count = 0

# a = []
# for i in coad_used:
#     if i[0:12] in apc:
#         count += 1
#         a.append(i)

# count = 0
# b = []
# for i in coad_used:
#     if i[0:12] in apc2:
#         count += 1
#         if i not in a:
#             b.append(i)
# print(count)
# c = list(set(a).union(set(b)))
# print(len(c))
# coad

# Add all patient with molecular profile variation information
# Prepare for saving into a json file.
coadread = []
for name in names:
    gene_info = [name]
    if name in rbfox1:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in wwox:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in CCSER1:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in CSMD1:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in TM9SF4:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in MACROD2:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in HCK:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in PLAGL2:
        gene_info.append(1)
    else:
        gene_info.append(0) 
    if name in TPX2:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in POFUT1:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in TSPY26P:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in FOXS1:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in ID1:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in KIF3B:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in REM1:
        gene_info.append(1)
    else:
        gene_info.append(0)  
    if name in PDRG1:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in COX4I2:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in MYLK2:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in DEFB118:
        gene_info.append(1)
    else:
        gene_info.append(0)
    if name in DUSP15:
        gene_info.append(1)
    else:
        gene_info.append(0)

    coadread.append(gene_info)

# same label infomation into json file
import csv

fields = ['id','CCSER1', 'COX4I2', 'CSMD1', 'DEFB118', 'DUSP15', 'FOXS1', 'HCK', 'ID1', 'KIF3B', 'MACROD2', 'MYLK2', 'PDRG1', 'PLAGL2', 'POFUT1', 'RBFOX1', 'REM1', 'TM9SF4', 'TPX2', 'TSPY26P', 'WWOX']

# data rows of csv file 
  
with open('top20_coadread_CNA.csv', 'w') as f:
      
    # using csv.writer method from CSV package
    write = csv.writer(f)
      
    write.writerow(fields)
    write.writerows(coadread)
