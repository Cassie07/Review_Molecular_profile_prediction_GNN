# Prediction on TCGA dataset


## 1. (On TCGA-COAD) Gene mutation and CNA : model training and prediction
* We use 10 fold cross-validation on TCGA-COAD dataset.

    ```
    CUDA_VISIBLE_DEVICES=4,5,6,7 python 2-1.coad_cross_validation_all_task.py
    ```

## 2. Validation model on TCGA-READ </br>【 Please define the task type before run the code 】

* We train the model on TCGA-COAD dataset, and validate model on TCGA-READ datast.
* This code could be used for all tasks, including gene mutation, CNA, MSI, and protein outcomes.


    ```
    CUDA_VISIBLE_DEVICES=4,5,6,7 python 2-2.read_validation_all_task.py
    ```
