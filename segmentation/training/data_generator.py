import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2
import os
import albumentations
import augmentations

def get_image(dirpath, image_filename, image_list, mask=False):
    
    if len(image_list) == 0:
        raise ValueError("Received Empty image list")

    # get extention of image
    imgext = image_list[0].split(".")[-1]

    if mask:
        # assuming extention of masks is png
        image_filename = image_filename.split(".")[0] + ".png"

    img_path = os.path.join(dirpath, image_filename)
    if img_path.split("/")[-1].split(".")[0] + f".{imgext}" not in image_list:
        print(image_list, "---------------- NOT FOUND -----------------------")
        print(img_path.split('/')[-1])
        raise ValueError('Incorrect Filepath, File not found')
    
    return img_path

def read_input_image(image_path):
    return cv2.resize(cv2.imread(image_path, 1), (32,32), interpolation=cv2.INTER_LINEAR)

def read_mask(mask_path):
    return cv2.resize(cv2.imread(mask_path, 0), (32,32), interpolation=cv2.INTER_LINEAR)

def get_processed(dirpath, image_list, mask=False, n_classes=2):
    image_array = []
    for img_filename in image_list:
        if not mask:
            img_path = get_image(dirpath, img_filename, image_list)
            # use rgb
            img = read_input_image(img_path)
        else:
            img_path = get_image(dirpath, img_filename, image_list, mask=True)
            img = read_mask(img_path)
            img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] // 255

        image_array.append(img)
    
    image_array = np.array(image_array)
    if mask:
        image_array = np.expand_dims(image_array, axis=-1)
        image_array = to_categorical(image_array, num_classes=n_classes)

    
    return (image_array)

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size, rand_aug=False, val=False, n_classes=2):
    L = len(img_list)

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            if not rand_aug:
                X = get_processed(img_dir, img_list[batch_start:limit])
                Y = get_processed(mask_dir, mask_list[batch_start:limit], mask=True)
            else:
                images = []
                masks = []
                for item in img_list[batch_start:limit]:
                    image = read_input_image(get_image(img_dir, item, img_list))
                    mask = read_mask(get_image(mask_dir, item, mask_list, mask=True))
                    if not val:
                        aug_out = augmentations.randAugmentation(image, mask)
                        images.append(aug_out['image'])
                        masks.append(cv2.threshold(aug_out['mask'], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] // 255)
                    else:
                        images.append(image)
                        masks.append(cv2.threshold(np.uint8(mask), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] // 255)
                    
                X = np.array(images)
                Y = np.array(masks)
                Y = to_categorical(np.expand_dims(Y, axis=-1), num_classes=n_classes)
            
            yield(X, Y)

            batch_start+=batch_size
            batch_end+=batch_size

