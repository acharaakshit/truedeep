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

def get_images(image_id, test_directory, size):
    train_images = []

    for directory_path in glob.glob(test_directory + 'images'):
        for img_path in set(sorted(glob.glob(os.path.join(directory_path, "*.jpg")))):
            if img_path.split("/")[-1] == image_id:
                img = cv2.imread(img_path, 1)
                if size is None:
                    # keep same size
                    size = (img.shape[1], img.shape[0])
                img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_LINEAR)
                train_images.append(img)
    train_images = np.array(train_images)
    return train_images