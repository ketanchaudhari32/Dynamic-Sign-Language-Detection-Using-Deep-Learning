#import necessary libraries
import cv2
import numpy as np
import glob
from multiprocessing import Pool, Process,  cpu_count

def scale(video, ratio):
    nb_return_frame = np.floor(ratio * len(video)).astype(int)
    return_ind = [int(i) for i in np.linspace(1, len(video), num=nb_return_frame)]

    return np.array([video[i-1] for i in return_ind])


def centre_crop(video, crop_percent):
    im_h, im_w, im_c = video[0].shape
    crop_h = int(im_h * (1-crop_percent))
    crop_w = int(im_h * (1-crop_percent))

    w1 = int(round((im_w - crop_w) / 2.))
    h1 = int(round((im_h - crop_h) / 2.))

    return np.array([img[h1:h1 + crop_h, w1:w1 + crop_w, :] for img in video])

def save_video(video,path):
    im_h, im_w, im_c = video[0].shape

    fps = 25

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (im_w,im_h))

    for frames in video:
        out.write(frames)
 
    out.release()


def data_augment(files, num_class,num_fps):
    for file in files:
        files_path = file.split('/')
        split = files_path[2]

        file_name = files_path[3].split('.')[0]

        cap = cv2.VideoCapture(file)

        video_buf = []


        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                video_buf.append(frame)

            else:
                break

        cap.release()

        video_buf = np.array(video_buf)

        
        #video scaling and resizing
        for i in [0.05,0.1,0.15,0.2,0.25]:#croping rates
            for j in [0.5, 0.8, 1.2, 1.5, 2.0]:#scaling ratio
                resized = centre_crop(video_buf, i)
                scaled = scale(resized, j)

                save_video(scaled,'./class_{}_dataset/{}/{}_crop_{}_scale_{}.mp4'.format(num_class,split,file_name,int(i*100),int(j*10)))


def parallel_process():
    cpu_cores = cpu_count()

    num_class = 10
    num_fps = 50
    for phase in ['train','val']:
        file_list = glob.glob('./class_{}_dataset/{}/*.mp4'.format(num_class,phase))
        # Split the input file list into sublists according to the number
        # of the available CPU cores
        for i in range(0, cpu_cores):
            sub_list = [file_list[j] for j in range(0, len(file_list)) if j % cpu_cores == i]

            if len(sub_list) > 0:
                p = Process(target=data_augment, args=([sub_list,num_class,num_fps]))
                p.start()

if __name__ == "__main__":
    parallel_process()