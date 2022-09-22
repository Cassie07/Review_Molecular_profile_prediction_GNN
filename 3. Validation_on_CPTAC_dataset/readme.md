# Validation on CPTAC dataset


## 1. Validation model on CPTAC-COAD <\br>【Please define the task type before run the code 】

* We train the model on TCGA-COAD dataset, and validate model on CPTAC-COAD datast.
* This code could be used for predicting gene mutation, CNA, and MSI outcomes.




    ```
    CUDA_VISIBLE_DEVICES=4,5,6,7 python 3-1.cptac_coad_validation_all_task.py
    ```
