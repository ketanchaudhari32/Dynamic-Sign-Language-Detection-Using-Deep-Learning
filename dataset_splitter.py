#import necessary libraries
import json
import os
import shutil
import numpy as np

#loading meta data file for WLASL dataset
class_labels = open("./dataset/WLASL_v0.3.json")
class_labels = json.load(class_labels)


#folders to store num class_data
num_class = 10
if not os.path.isdir('./class_{}_dataset'.format(num_class)):
  os.mkdir('./class_{}_dataset'.format(num_class))

#create train, test and val folder
if not os.path.isdir('./class_{}_dataset/train'.format(num_class)):
  os.mkdir('./class_{}_dataset/train'.format(num_class))

if not os.path.isdir('./class_{}_dataset/test'.format(num_class)):
  os.mkdir('./class_{}_dataset/test'.format(num_class))

if not os.path.isdir('./class_{}_dataset/val'.format(num_class)):
  os.mkdir('./class_{}_dataset/val'.format(num_class))


class_labels_used = [] #array to store used labels names

#itearting over WLASL meta file to get location of classes and move to new folder created
for i in range(0, num_class):
  class_name = class_labels[i]['gloss']
  class_labels_used.append(class_name)
  #interating over all availble files for current class
  for j in range(len(class_labels[i]['instances'])):
    video_id = class_labels[i]['instances'][j]['video_id'] #file name of video file
    split = class_labels[i]['instances'][j]['split'] #get split i.e train/test/val
    if os.path.exists('./dataset/videos/{}.mp4'.format(video_id)): #checking if file location exist in video dataset folder
      #copying file from original dataset location to new folder created loacation
      shutil.copy('./dataset/videos/{}.mp4'.format(video_id),'./class_{}_dataset/{}/{}_{}.mp4'.format(num_class,split,class_name,j))

#storing class labels as numpy array for futher reference
class_labels_used = np.array(class_labels_used)
np.save('./class_{}_dataset/class_labels.npy'.format(num_class),class_labels_used)


