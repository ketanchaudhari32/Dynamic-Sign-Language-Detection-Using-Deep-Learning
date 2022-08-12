# Dynamic-Sign-Language-Detection-Using-Deep-Learning

Pre-requisite:
1.  Download the dataset from the following link:    
[WLASL dataset](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)
2. Extract the folder and transfer the 
1.  Create python virtual environment:    
```conda create --name <env_name> --file requirements.txt```
2.  Activate virual env:    
```conda activate <env_name>```


Files Walkthrough:    
1.  dataset_splitter.py: This file read the original dataset and create folder specific num_classes containing training, val and test data. And copies files from the original dataset folder to num_classes data folder.

2.  data_augment.py: This file parse through training and validation data and creates copies of files with differnet crop ratio and scaling rate.

3. 