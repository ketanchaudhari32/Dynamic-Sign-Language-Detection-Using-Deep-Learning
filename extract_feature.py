#importing libraries
import cv2
import numpy as np
import os
import mediapipe as mp
import glob
from multiprocessing import Pool, Process,  cpu_count

#folders to store features
num_class = 10

if not os.path.isdir('./class_{}_features'.format(num_class)):
  os.mkdir('./class_{}_features'.format(num_class))

#create train, test and val folder
if not os.path.isdir('./class_{}_features/train'.format(num_class)):
  os.mkdir('./class_{}_features/train'.format(num_class))

if not os.path.isdir('./class_{}_features/test'.format(num_class)):
  os.mkdir('./class_{}_features/test'.format(num_class))

if not os.path.isdir('./class_{}_features/val'.format(num_class)):
  os.mkdir('./class_{}_features/val'.format(num_class))


#mediapipe helper function to read each frame and provide features
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


#Extracting features for each keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33,3))
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468,3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21,3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21,3))

    #padding zeros at end of array to make dimenion eqal to convert into square matrix
    feature_array = np.concatenate([pose, face, lh, rh, np.zeros((33,3))]) 
    feature_array = feature_array.reshape((24,24,3)) #reshapping into square matrix

    return feature_array

#helper function to read videos files and extract features
def extract_feature(files):
    #iterating over file list
    for file in files:
        #getting file name
        files_path = file.split('/')
        split = files_path[2]
        file_name = files_path[3].split('.')[0]

        cap = cv2.VideoCapture(file) #reading video file

        mp_holistic = mp.solutions.holistic # Using holistic model of mediapip framework

        results_array = [] #array to store individual frame features

        #reading frames from video and extracting features
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    image, results = mediapipe_detection(frame, holistic)
                    keypoints = extract_keypoints(results)
                    results_array.append(keypoints)

                else:
                    break

            #closing video reader
            cap.release()
            cv2.destroyAllWindows()
            
        results_array = np.array(results_array) #converting result to array

        # array_len = len(results_array)
        avg_frame_count = 100 #setting number of frames to be trained model with

        #scalling video to threshold frames length
        return_ind = [int(i) for i in np.linspace(1, len(results_array), num=avg_frame_count)]
        results_array = [ results_array[i-1] for i in return_ind]
    
        #saving the extracted features
        np.save('./class_{}_features/{}/{}.npy'.format(num_class,split,file_name),results_array)
        
#function to reading file using parallel processing      
def parallel_process():
    cpu_cores = cpu_count() #getting number of cpu cores available

    #checking if features are already existed for current video
    existing_features = glob.glob('./class_{}_features/*/*'.format(num_class,))
    for i in range(len(existing_features)):
        existing_features[i] = existing_features[i].replace('.npy','.mp4')
        existing_features[i] = existing_features[i].replace('_features','_dataset')
    current_list = glob.glob('./class_{}_dataset/*/*.mp4'.format(num_class))
    file_list = list(set(current_list) - set(existing_features))
    
    # Split the input file list into sublists according to the number
    # of the available CPU cores
    for i in range(0, cpu_cores):
        sub_list = [file_list[j] for j in range(0, len(file_list)) if j % cpu_cores == i]

        if len(sub_list) > 0:
            p = Process(target=extract_feature, args=([sub_list]))
            p.start()

if __name__ == "__main__":
    parallel_process()#creating parallel process to reduce computation time

            


