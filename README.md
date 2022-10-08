# Overview

We acknowledge that the potential variance of reproducing results could exist due to the various data preprocessing, label collection, and patch (e.g., graph node) selections.

This repository contains the code for the following manuscript:

Spatially-aware Graph Neural Networks Enable Cross-level Molecular Profile Prediction in Colon Cancer Histopathology: A Retrospective Multicentre Cohort Study, submitted to <i>The LANCET Digital Health</i>.


# Repo Structure

```
├── functions
    ├── model_architecture.py
    ├── model_utils.py
    └── utils.py
├── 1. Data_preprocessing
    ├── 1-1-1.prepare_molecular_label.py
    ├── 1-2-1.segment_tiles.py
    ├── 1-2-2.tumor_detection.py
    ├── 1-2-3.tumor_prediction.py
    ├── 1-3-1.tile_normalization.py
    ├── 1-3-2.feature_extraction.py
    ├── 1-4-1.generate_adj_similar.py
    └── readme.md
├── 2. Prediction_on_TCGA_dataset
    ├── 2-1.coad_cross_validation_all_task
    ├── 2-2.read_validation_all_task
    └── readme.md
├── 3. Validation_on_CPTAC_dataset
    ├── 3-1.cptac_coad_validation_all_task
    ├── 2-2.read_validation_all_task
    └── readme.md
```

# Dependencies

```
Pytorch 1.6.0

torch-geometric 1.6.1

Torchvision 0.7.0

openslide 1.1.1

Pillow 6.2.0

numpy 1.16.4

pandas 0.25.1

scikit-image 0.15.0

scikit-learn 0.21.3

Pillow 6.2.0

h5py 2.8.0

```

# Data download

Download the FFPE whole slide images from GDC portal (https://portal.gdc.cancer.gov/) for colon adenocarcinoma (TCGA-COAD), rectum adenocarcinoma (TCGA-READ), and CANCER portal (https://wiki.cancerimagingarchive.net/display/Public/CPTAC-COAD) colon adenocarcinomacolon adenocarcinoma (CPTAC-COAD).

Download corresponding omics data (gene mutation, cna, expression) from cBioPortal (https://www.cbioportal.org/) for each dataset.


# Usage

## Step 1. Data preprocessing

Using the code under `1. Data_preprocessing` to perform


### 1. Patch extraction and molecular profile label preparation. 
* Get gene mutation, CNA, MSI, and protein information of selected slides
    ```
    python 1-1-1.prepare_molecular_label.py
    ```
 
### 2. Patch extraction and selection
* Extract patches
    ```
    python 1-2-1.segment_tiles.py
    ```
* Tumor detection model training
    ```
    python 1-2-2.tumor_detection.py
    ```
* Tumor patch prediction (selection by tumor detection model)
    ```
    python 1-2-3.tumor_prediction.py
    ```

### 3. Color normmalization and feature extraction
* Color normalization
    ``` 
    python 1-3-1.tile_normalization.py
    ```
* Extract features for each patch
    ```
    python 1-3-2.feature_extraction.py
    ```
    
### 4. Graph construction
* Generate adjacency matrix infomation (construct graph edge). For each node, we save its connected neighbor nodes.
    ``` 
    python 1-4-1.generate_adj_similar.py
    ```


## Step 2. Model training for molecular profile prediction on TCGA-COAD, and validating on TCGA-READ

Using the code under `2. Prediction_on_TCGA_dataset` 

<!-- #### Prediction on TCGA dataset -->


### 1. (On TCGA-COAD) Gene mutation and CNA : model training and prediction
* We use 10 fold cross-validation on TCGA-COAD dataset.
* This code could be used for all tasks, including gene mutation, CNA, MSI, and protein.
    ```
    CUDA_VISIBLE_DEVICES=1,2,3,4 python 2-1.coad_cross_validation_all_task.py
    ```


### 2. Validation model on TCGA-READ </br>【 Please define the task type before run the code 】

* We train the model on TCGA-COAD dataset, and validate model on TCGA-READ datast.
* This code could be used for all tasks, including gene mutation, CNA, MSI, and protein.

    ```
    CUDA_VISIBLE_DEVICES=1,2,3,4 python 2-2.read_validation_all_task.py
    ```


## Step 3. Model validation for molecular profile prediction on CPTAC-COAD

Using the code under `3. Validation_on_CPTAC_dataset` 

<!-- #### Validation on CPTAC dataset -->


### 1. Validation model on CPTAC-COAD </br>【Please define the task type before run the code 】

* We train the model on TCGA-COAD dataset, and validate model on CPTAC-COAD datast. 
* This code could be used for predicting gene mutation, CNA, and MSI. 


    ```
    CUDA_VISIBLE_DEVICES=1,2,3,4 python 3-1.cptac_coad_validation_all_task.py
    ```
    
# Citation

```
```


