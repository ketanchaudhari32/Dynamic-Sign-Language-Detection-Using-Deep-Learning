#import necessary libraries
import cv2
import numpy as np
import glob
from multiprocessing import Pool, Process,  cpu_count

#fuuntion to scale the frame length
#i.e upscaling and downscaling based on scaling ratio
def scale(video, ratio):
    nb_return_frame = np.floor(ratio * len(video)).astype(int) #get number of frames length after scaling
    return_ind = [int(i) for i in np.linspace(1, len(video), num=nb_return_frame)] #scale the frames
    return np.array([video[i-1] for i in return_ind]) #return array of new frame sequence


#function to crop video
def centre_crop(video, crop_percent):
    im_h, im_w, im_c = video[0].shape #get dimensions of video

    #new dimesnion after cropping
    crop_h = int(im_h * (1-crop_percent)) 
    crop_w = int(im_h * (1-crop_percent))
    w1 = int(round((im_w - crop_w) / 2.))
    h1 = int(round((im_h - crop_h) / 2.))

    return np.array([img[h1:h1 + crop_h, w1:w1 + crop_w, :] for img in video]) #return cropped video

#function to save video
def save_video(video,path):
    im_h, im_w, im_c = video[0].shape #get dimension of video from frame generated

    fps = 25 #initialising frames per second to save the video

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #setting format to save video
    out = cv2.VideoWriter(path, fourcc, fps, (im_w,im_h)) #video writer based on intialise parameters

    #writing frames to video writer
    for frames in video:
        out.write(frames)
 
    out.release() #closing the writer


#function to call video augmention operation
def data_augment(files, num_class):
    #iterating over list of files
    for file in files:
        #getting file name
        files_path = file.split('/') 
        split = files_path[2]
        file_name = files_path[3].split('.')[0]

        #reading the video file
        cap = cv2.VideoCapture(file)

        video_buf = []#array to store frames of video

        #iterating while input is true
        while(cap.isOpened()):
            ret, frame = cap.read() #reading frames
            if ret:
                video_buf.append(frame) #appending frames to buffer array
            else:
                break

        cap.release() #closing video reader

        video_buf = np.array(video_buf)
        
        #video scaling and resizing
        for i in [0.05,0.1,0.15,0.2,0.25]:#croping rates
            for j in [0.5, 0.8, 1.2, 1.5, 2.0]:#scaling ratio
                resized = centre_crop(video_buf, i) #calling cropping function
                scaled = scale(resized, j) #caller scalling function

                #saving augmented video
                save_video(scaled,'./class_{}_dataset/{}/{}_crop_{}_scale_{}.mp4'.format(num_class,split,file_name,int(i*100),int(j*10)))

#function to reading file using parallel processing
def parallel_process():
    cpu_cores = cpu_count() #getting number of cpu cores available

    num_class = 10 #number of classes to be trained on

    #augmenting only training and validation data
    for phase in ['train','val']: 
        file_list = glob.glob('./class_{}_dataset/{}/*.mp4'.format(num_class,phase))
        # Split the input file list into sublists according to the number
        # of the available CPU cores
        for i in range(0, cpu_cores):
            sub_list = [file_list[j] for j in range(0, len(file_list)) if j % cpu_cores == i]

            if len(sub_list) > 0:
                p = Process(target=data_augment, args=([sub_list,num_class]))
                p.start()

if __name__ == "__main__":
    parallel_process()#creating parallel process to reduce computation time