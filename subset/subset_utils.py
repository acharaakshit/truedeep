import glob
import os
import cv2
import numpy as np

def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    files = []
    for file in listOfFile:
        filename =os.fsdecode(file)
        files.append(filename)
    return files

def get_images(image_id, test_directory):
    train_images = []

    for directory_path in glob.glob(test_directory + 'images'):
        for img_path in set(sorted(glob.glob(os.path.join(directory_path, "*.jpg")))):
            if img_path.split("/")[-1] == image_id:
                img = cv2.resize(cv2.imread(img_path, 1), dsize=(32,32), interpolation=cv2.INTER_LINEAR)
                train_images.append(img)
    train_images = np.array(train_images)
    return train_images