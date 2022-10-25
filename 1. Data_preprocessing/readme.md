# Data processing steps


## Step 1. Patch generation and molecular profile label preparation. 
* Get gene mutation, CNA, MSI, and protein information of selected slides
    ```
    python 1-1.prepare_molecular_label.py
    ```
 
 
## Step 2. Patch extraction and selection
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

## Step 3. Color norm and feature extraction
* Color normalization
    ``` 
    python 3-1.tile_normalization.py
    ```
* Extract features for each patch
    ```
    python 3-2.feature_extraction.py
    ```
    
## Step 4. Graph construction
* Generate adjacency matrix infomation (construct graph edge). For each node, we save its connected neighbor nodes.
    ``` 
    python 4-1.generate_adj_similar.py
    ```

```
