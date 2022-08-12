# Dynamic-Sign-Language-Detection-Using-Deep-Learning

Pre-requisite:
1.  Download the dataset from the following link:    
[WLASL dataset](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)
2. Extract the datset and transfer the videos folder to dataset folder in current base directory containing these code files.
3.  Create python virtual environment:    
```conda create --name <env_name> --file requirements.txt```
4.  Activate virual env:    
```conda activate <env_name>```
5.  Download the trained model from the link and move to trained_model folder:        
[Trained Model](https://qmulprod-my.sharepoint.com/:u:/g/personal/ec21208_qmul_ac_uk/ESclUM-b0ZBMov3h41iRGu8BRjp-RWEuK6dOxHKhzrmJjQ?e=14g8xF)

Files Walkthrough with order of operations:    
1.  dataset_splitter.py: This file read the original dataset and create folder specific num_classes containing training, val and test data. And copies files from the original dataset folder to num_classes data folder.

2.  data_augment.py: This file parse through training and validation data and creates copies of files with differnet crop ratio and scaling rate.

3. extract_features.py: This file uses media_pip framework to extract 3d features from the video files

4. spatio_temporal_conv.py: Code containing class of R(2+1)D CNN model.

5. sign_language_real_time.py: Code to do real time sign language detection.
