import spams
import cv2 as cv
import os
import numpy as np
from PIL import Image
import random

folder = 'fold_segmented tiles before normalization'

# load patch names
name = './tumor tiles name'.format(folder)
with open(name) as f:
    patch_name = [line.split('\n')[0] for line in f if not line.startswith('.')]

# load patch predicted result
name = './tumor tiles lable'.format(folder)
with open(name) as f:
    patch_label = [int(line.split('\n')[0]) for line in f if not line.startswith('.')]

nonorm_patch = []
for name,label in zip(patch_name, patch_label):
    if label == 1:
        nonorm_patch.append(name)

print('Total patches : {}'.format(len(patch_name)))
print('Tumor patches : {}'.format(len(nonorm_patch)))
print(nonorm_patch[0])
print()

image_path = './{}'.format(folder)
save_path = './normalization_patches'

# load image
def read_image(path):
    im = cv.imread(path)
    # Convert from cv2 standard of BGR to our convention of RGB.
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return im

class TissueMaskException(Exception):
    pass

######################################################################################################

def is_uint8_image(I):
    if not is_image(I):
        return False
    if I.dtype != np.uint8:
        return False
    return True
######################################################################################################

def is_image(I):
    if not isinstance(I, np.ndarray):
        return False
    if not I.ndim == 3:
        return False
    return True
######################################################################################################

def get_tissue_mask(I, luminosity_threshold=0.8):
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
    mask = L < luminosity_threshold

    # Check it's not empty
    if mask.sum() == 0:
        raise TissueMaskException("Empty tissue mask computed")

    return mask

######################################################################################################

def convert_RGB_to_OD(I):
    mask = (I == 0)
    I[mask] = 1


    #return np.maximum(-1 * np.log(I / 255), 1e-6)
    return np.maximum(-1 * np.log(I / 255), np.zeros(I.shape) + 0.1)

######################################################################################################

def convert_OD_to_RGB(OD):

    assert OD.min() >= 0, "Negative optical density."

    OD = np.maximum(OD, 1e-6)

    return (255 * np.exp(-1 * OD)).astype(np.uint8)

######################################################################################################

def normalize_matrix_rows(A):
    return A / np.linalg.norm(A, axis=1)[:, None]

######################################################################################################


def get_concentrations(I, stain_matrix, regularizer=0.01):
    OD = convert_RGB_to_OD(I).reshape((-1, 3))
    return spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True).toarray().T

######################################################################################################

def get_stain_matrix(I, luminosity_threshold=0.8, angular_percentile=99):

    # Convert to OD and ignore background
    tissue_mask = get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,))
    OD = convert_RGB_to_OD(I).reshape((-1, 3))

    OD = OD[tissue_mask]

    # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))

    # The two principle eigenvectors
    V = V[:, [2, 1]]

    # Make sure vectors are pointing the right way
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1

    # Project on this basis.
    That = np.dot(OD, V)

    # Angular coordinates with repect to the prinicple, orthogonal eigenvectors
    phi = np.arctan2(That[:, 1], That[:, 0])

    # Min and max angles
    minPhi = np.percentile(phi, 100 - angular_percentile)
    maxPhi = np.percentile(phi, angular_percentile)

    # the two principle colors
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    # Order of H and E.
    # H first row.
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])

    return normalize_matrix_rows(HE)

######################################################################################################

def mapping(target,source):

    stain_matrix_target = get_stain_matrix(target)
    target_concentrations = get_concentrations(target,stain_matrix_target)
    maxC_target = np.percentile(target_concentrations, 99, axis=0).reshape((1, 2))
    stain_matrix_target_RGB = convert_OD_to_RGB(stain_matrix_target)

    stain_matrix_source = get_stain_matrix(source)
    source_concentrations = get_concentrations(source, stain_matrix_source)
    maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
    source_concentrations *= (maxC_target / maxC_source)
    tmp = 255 * np.exp(-1 * np.dot(source_concentrations, stain_matrix_target))
    return tmp.reshape(source.shape).astype(np.uint8)

# load normalization reference image
print('starting')
source = cv.imread('Ref.png')
#print(source)
source = cv.cvtColor(source, cv.COLOR_BGR2RGB)

# Normalize all segmented tiles
transform_list=[]
count = 1
for i in nonorm_patch:
    if not i.startswith('.'):
        #name = i[46:]
        #print(name)
        path = image_path + '/' + i
        target = cv.imread(path)
#        print(name)
        #print(path)
        target = cv.cvtColor(target, cv.COLOR_BGR2RGB)
        transformed = mapping(source, target)
        #print(transformed)
        new_im=np.array(transformed)
        new_im=Image.fromarray(new_im)
        new_im=new_im.resize((224,224),Image.ANTIALIAS)
        
        # save normalized tiles
        new_im.save(save_path + '/'+ i)
        transform_list.append(save_path + '/'+ i)
        print('finish {}'.format(count))
        count += 1
        #print(name)
        
# save normalized patch names  
with open('./{}_normalized_saving_path_name.txt'.format(folder), 'a') as f:
   for item in transform_list:
       f.write("%s\n" % item)

print('finish written')
