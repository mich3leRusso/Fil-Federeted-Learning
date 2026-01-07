import numpy as np 
import cv2
import os
import glob
import shutil


## IMPORTANT READ THIS BEFORE RUNNING THIS SCRIPT ###


PATH_DATASET = '/davinci-1/home/dmor/PycharmProjects/MIND/data_128x128/core50_128x128/*'


shutil.copytree(PATH_DATASET[:-1], '/davinci-1/home/dmor/PycharmProjects/MIND/data_64x64/core50_64x64/')

list_folder = glob.glob(PATH_DATASET)
print(list_folder, len(list_folder))
#import ipdb; ipdb.set_trace()
for folder in list_folder:
    list_folder_L1 = glob.glob(folder+'/*')
    #print(list_folder_L1)
    for l1 in list_folder_L1:
        print(l1)
        imgs = glob.glob(l1+'/*.png')
        #print(imgs)
        for img_path in imgs:
            #print('passato')
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            resized = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)
            #print(img_path)
            img_path_changed = img_path.replace('_128x128','_64x64')
            #print(img_path_changed)
            cv2.imwrite(img_path_changed, resized)

