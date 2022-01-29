# Overview of Molecualr profile prediction in colon cancer

The code in this repository is confidential and for review purpose only. The link to access the code (anyone with the link could access the code): 

The link is shared in data sharing section.

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

## Data download

Download the FFPE whole slide images from GDC portal (https://portal.gdc.cancer.gov/) for colon adenocarcinoma (TCGA-COAD), rectum adenocarcinoma (TCGA-READ), and CANCER portal (https://wiki.cancerimagingarchive.net/display/Public/CPTAC-COAD) colon adenocarcinomacolon adenocarcinoma (CPTAC-COAD).

Download corresponding omics data (gene mutation, cna, expression) from cBioPortal (https://www.cbioportal.org/) for each dataset.


## Usage

### Step 1. Data preprocessing

Using the code under `1. Data_preprocessing` to perform


#### 1. Patch generation and molecular profile label preparation. 
* Get gene mutation, CNA, MSI, and protein information of selected slides
    ```
    python 1-1.prepare_molecular_label.py
    ```
 
#### 2. Patch extraction and selection
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

#### 3. Color norm and feature extraction
* Color normalization
    ``` 
    python 3-1.tile_normalization.py
    ```
* Extract features for each patch
    ```
    python 3-2.feature_extraction.py
    ```
    
#### 4. Graph construction
* Generate adjacency matrix infomation (construct graph edge). For each node, we save its connected neighbor nodes.
    ``` 
    python 4-1.generate_adj_similar.py
    ```



### Step 2. Model training for molecular profile prediction on TCGA-COAD, and validating on TCGA-READ

Using the code under `2. Prediction_on_TCGA_dataset` 

#### Prediction on TCGA dataset


#### 1. (On TCGA-COAD) Gene mutation and CNA : model training and prediction
* We use 10 fold cross-validation on TCGA-COAD dataset.

    ```
    python 1.coad_cross_validation_gene_or_cna.py.py
    ```
#### 2. (On TCGA-COAD) MSI : model training and prediction
* We use 10 fold cross-validation on TCGA-COAD dataset.

    ```
    python 2.coad_cross_validation_msimss.py
    ```
    
#### 3. (On TCGA-COAD) Protein : model training and prediction
* We use 10 fold cross-validation on TCGA-COAD dataset.

    ```
    python 3.coad_cross_validation_protein.py
    ```

#### 4. Validation model on TCGA-READ </br>【 Please define the task type before run the code 】

* We train the model on TCGA-COAD dataset, and validate model on TCGA-READ datast.
* This code could be used for all tasks, including gene mutation, CNA, MSI, and protein outcomes.

    ```
    python 4.read_validation_all_task.py
    ```


### Step 3. Model validation for molecular profile prediction on CPTAC-COAD

Using the code under `3. Validation_on_CPTAC_dataset` 

#### Validation on CPTAC dataset


#### 1. Validation model on CPTAC-COAD </br>【Please define the task type before run the code 】

* We train the model on TCGA-COAD dataset, and validate model on CPTAC-COAD datast. 
* This code could be used for predicting gene mutation, CNA, and MSI outcomes. 


    ```
    python 1.cptac_read_validation_all_task.py
    ```


