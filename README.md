# Molecualr profile prediction in colon cancer

The code in this repository is confidential and for review purpose only.

This repository contains the code for the following manuscript:

Spatially-aware Graph Neural Networks Enable Cross-level Molecular Profile Prediction in Colon Cancer Histopathology: A Retrospective Multicentre Cohort Study, submitted to <i>The LANCET Digital Health</i> for review.



## Dependencies

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


## Usage

### Step 1. Data download

Download the FFPE whole slide images from GDC portal (https://portal.gdc.cancer.gov/) for colon adenocarcinoma (TCGA-COAD), rectum adenocarcinoma (TCGA-READ), and CANCER portal (https://wiki.cancerimagingarchive.net/display/Public/CPTAC-COAD) colon adenocarcinomacolon adenocarcinoma (CPTAC-COAD).

Download corresponding omics data (gene mutation, cna, expression) from cBioPortal (https://www.cbioportal.org/) for each dataset.

### Step 2. Data preprocessing

Using the code under `1. Data_preprocessing` to perform

# Data processing steps


## 1. Patch generation and molecular profile label preparation. 
* Get gene mutation, CNA, MSI, and protein information of selected slides
    ```
    python 1-1.prepare_molecular_label.py
    ```
 
## 2. Patch extraction and selection
* Extract patches
    ```
    python 2-1.segment_tiles.py
    ```
* Tumor detection model training
    ```
    python 2-2.tumor_detection.py
    ```
* Tumor patch prediction (selection by tumor detection model)
    ```
    python 2-3.tumor_prediction.py
    ```

## 3. Color norm and feature extraction
* Color normalization
    ``` 
    python 3-1.tile_normalization.py
    ```
* Extract features for each patch
    ```
    python 3-2.feature_extraction.py
    ```
    
## 4. Graph construction
* Generate adjacency matrix infomation (construct graph edge). For each node, we save its connected neighbor nodes.
    ``` 
    python 4-1.generate_adj_similar.py
    ```




### Step 3. Model training for molecular profile prediction on TCGA-COAD, and validating on TCGA-READ

Using the code under `2. Prediction_on_TCGA_dataset` 



### Step 4. Model validation for molecular profile prediction on CPTAC-COAD

Using the code under `3. Validation_on_CPTAC_dataset` 

```
