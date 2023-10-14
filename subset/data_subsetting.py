from ast import arg, literal_eval
import bisect
import glob
import math
import os
import pickle
import random
from tensorflow import keras
import segmentation_models as sm
from keras_unet_collection import models, base, utils, losses
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
from subset_utils import getListOfFiles, get_images
import cv2
import argparse
import shutil

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_model(backbone, n_classes):
    model = models.unet_2d((None, None, 3), filter_num=[16, 32, 64, 128, 256, 256],
            n_labels=n_classes,
            stack_num_down=2, stack_num_up=2,
            activation='ReLU',
            output_activation='Softmax',
            batch_norm=True, pool=True, unpool=True,
            backbone=backbone, weights='imagenet',
            freeze_backbone=False, freeze_batch_norm=False,
            name='unet')

    loss_fn = sm.losses.binary_focal_dice_loss
    model.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['accuracy', losses.dice_coef])
    return model

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def save_outputs(model, directory, backbone, size):
    selected_set = getListOfFiles(os.fsencode(directory + 'images'))
    random.Random(51).shuffle(selected_set)
    all_outputs = dict()
    for image in selected_set:
        X_test = get_images(image, directory, size)
        encoder = model.get_layer(backbone + "_backbone")
        output = encoder.predict(X_test)
        all_outputs[image] = np.concatenate([x.ravel() for x in output])
        print(image)
    
    return all_outputs

def cluster(all_outputs, json_path, n_components):

    v = list(all_outputs.values())
    img_name = list(all_outputs.keys())
    data = np.stack(v, axis=0)
    print(data.shape)

    mapping = dict()

    transformer = PCA(n_components=n_components)
    data_2d = transformer.fit_transform(data)

    print(data_2d.shape)

    for j in range(0, n_components):
        data_2d[:, j] = scale_to_01_range(data_2d[:, j])

    for i in range(0, len(v)):
        print(img_name[i])
        img_val = []
        for z in range(0, n_components):
            img_val.append(data_2d[:, z][i])

        mapping[img_name[i]] = str(tuple(img_val))

    with open(json_path, 'w+') as f:
        json.dump(mapping, f)

def save_true_map(point_map, selected_images_train, selected_image_val, true_mapping_path):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    selected_images = tuple((set(selected_images_train), set(selected_image_val)))
    for selected_image in selected_images:
        print(f"Selected: {selected_images}")
        points = []
        for selected in list(selected_image):
            points.append(point_map[selected])
        
        x_val = [x[0] for x in points]
        y_val = [x[0] for x in points]

        ax.scatter(x_val, y_val, s=6)
    
    plt.subplots_adjust(right=0.7)
    fig.savefig(true_mapping_path.split(".")[0] + ".png")
    plt.close(fig=fig)

def get_img_dist(point_map, true_mapping_path, directory, dims_to_take):

    for k, v in point_map.items():
        if type(v) == str:
            val = literal_eval(v)
            point_map[k] = val

    average = [sum(x)/ len(x) for x in zip(*list(point_map.values()))]
    print(average)
    all_selected_train = []
    all_selected_val = []
    toll = 1
    n_bins = 10
    for dimension in range(0, len(average)):
        if dimension == dims_to_take:
            break
        distances = []
        dimension_point_map = dict()

        for k, v in point_map.items():
            dist = 0
            dist += (v[dimension] - average[dimension])*(v[dimension] - average[dimension])
            dist = math.sqrt(dist)
            distances.append(dist)
            v_new = [v[dimension], dist]
            v_new = tuple(v_new)
            dimension_point_map[k] = v_new

        n, bins, patches = plt.hist(distances, bins=n_bins)
        print(n, bins)
        bin_mapping = dict()

        for b_idx in range(0, len(n)):
            bin_mapping[b_idx] = []

        for k, v in dimension_point_map.items():
            bin_index = bisect.bisect(bins, v[-1]) - 1
            if bin_index == len(bins) - 1 and bins[len(bins) - 1] == v[-1]:
                bin_index = len(bins) - 2
            bin_mapping[bin_index].append(k)

        folder = './dimension-' + str(dimension) + '/'
        os.makedirs(folder, exist_ok=True)
        files = glob.glob(folder+'*')
        for f in files:
            os.remove(f)

        cnt = 1
        for selimg in bin_mapping.values():
            for sel in selimg:
                a = cv2.imread(directory + 'images/' + sel, 1)
                cv2.imwrite(folder + sel.split('.')[0] + '_' + str(cnt) + '.jpg', a)
            cnt+=1
        
        with open(true_mapping_path, 'wb') as f:
            pickle.dump(bin_mapping, f)

        n_mapping = {}
        for ind in range(0, len(n)):
            n_mapping[ind] = n[ind]
        
        n_mapping = dict(sorted(n_mapping.items(), key=lambda item: item[1], reverse=True))

        s_percent = 0.5/toll
        dec = s_percent/(n_bins-1)

        toll=toll*8
        m=len(distances)
        min_take_mapping = dict()
        min_take = round((m*s_percent) / len(bins) - 1)

        for bin_idx, num_images in n_mapping.items():
            n_mapping[bin_idx] = s_percent
            min_take_mapping[bin_idx] = min_take
            s_percent -= dec
            min_take = round((m * s_percent) / (n_bins))
            if s_percent <= 0:
                s_percent = 0
                dec = 0
                min_take = 0
        
        print(n_mapping, min_take_mapping)

        selected_images_train = []
        selected_images_val = []

        for k,v in bin_mapping.items():
            val_clean_v = []
            train_set = []
            val_set = []

            xi = len(v)
            if xi == 0:
                continue

            min_take = min_take_mapping[k]

            if min_take > len(v):
                to_take = len(v)
            else:
                to_take = min_take

            print(f"The images in bin are {len(v)}")

            # 90-10 split
            train_take = round(to_take*0.90)

            val_take = to_take - train_take

            v = list(v)

            val_clean_v = v.copy()

            if val_take > 0:
                jump = round(len(v) / val_take)
                print(f"The value of val jump is {jump}")
                print(f"The images in bin {k} are {v}")
                for z in range(0, val_take):
                    val_idx = round((z*jump) + jump/2)
                    if val_idx >= len(v):
                        break
                    print(f"Taking validation index {val_idx} for length {len(v)}")
                    val_set.extend([v[val_idx]])
                    val_clean_v.remove(v[val_idx])
            
            if train_take > 0:
                jump = round(len(val_clean_v) / train_take)
                print(f"The value of train jump is {jump}")
                if jump == 1:
                    random.Random(51).shuffle(val_clean_v)
                    if len(val_clean_v) == 2:
                        train_take = 1
                    train_set = val_clean_v[:train_take]
                else:
                    for z in range(0, train_take):
                        train_idx = round((z*jump) + jump/2)
                        if train_idx >=  len(val_clean_v):
                            break
                        print(f"Taking train index {train_idx} for length {len(val_clean_v)}")
                        train_set.extend([val_clean_v[train_idx]])
            
            selected_images_train.extend(train_set)
            print(f"Picking {len(train_set)} images for train which are {train_set} for bin{k}")
            selected_images_val.extend(val_set)
            print(f"Picking {len(val_set)} images for validation which are {val_set} for bin {k}")
        
        n_bins = n_bins//2

        print(len(selected_images_train), selected_images_train)
        print(len(set(selected_images_train)), set(selected_images_train))
        print(len(selected_images_val), selected_images_val)
        print(len(set(selected_images_val)), set(selected_images_val))
        all_selected_train.extend(selected_images_train)
        all_selected_val.extend(selected_images_val)

    print("The overall selection is as follows")
    print(set(all_selected_train), set(all_selected_val))
    print(len(set(all_selected_train)), len(set(all_selected_val)))
    save_true_map(point_map, all_selected_train, all_selected_val, true_mapping_path)
    return (all_selected_train, all_selected_val)

def read_mapping(true_mapping_path, json_path, true_path, directory, dims_to_take):
    with open(json_path, 'r+') as f:
        mapping = json.load(f)
    
    t_images, v_images = get_img_dist(mapping, true_mapping_path, directory, dims_to_take)

    for image in t_images:
        img_path = os.path.join(directory, 'images/', image)
        msk_path = os.path.join(directory, 'masks/', image)
        img = cv2.imread(img_path, 1)
        msk = cv2.imread(msk_path.split('.')[0] + '.png', 0)

        cv2.imwrite(os.path.join(true_path, 'train/images/', image), img)
        cv2.imwrite(os.path.join(true_path, 'train/masks/', image.split('.')[0] + '.png'), msk)

    for image in v_images:
        img_path = os.path.join(directory, 'images/', image)
        msk_path = os.path.join(directory, 'masks/', image)

        img = cv2.imread(img_path, 1)
        msk = cv2.imread(msk_path.split('.')[0] + '.png', 0)

        cv2.imwrite(os.path.join(true_path, 'val/images/', image), img)
        cv2.imwrite(os.path.join(true_path, 'val/masks/', image.split('.')[0] + '.png'), msk)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default="EfficientNetB0")
    parser.add_argument('--directory', default='data/train')
    parser.add_argument('--json_path', default='json/image_point_map.json')
    parser.add_argument('--true_mapping_path', default='plots/trueset.png')
    parser.add_argument('--true_path', default='target/')
    parser.add_argument('--image_size', nargs='+', type=int, default=None, 
                    help="""The image size to be used for getting the 
                    feature representation, target images will be saved 
                    in the initial size. The input should be in the 
                    format: height<space>width""")

    args = parser.parse_args()

    backbone = args.backbone
    model = get_model(backbone, n_classes=2)
    directory = args.directory
    # json file to store the image-coordinate mapping
    json_path = args.json_path
    os.makedirs("/".join(json_path.split("/")[:-1]), exist_ok=True)
    # trueset representation of images from first dimension of PCA
    true_mapping_path = args.true_mapping_path
    os.makedirs("/".join(true_mapping_path.split("/")[:-1]), exist_ok=True)
    true_path = args.true_path
    # clean the existing target folder
    if os.path.isdir(true_path):
        shutil.rmtree(true_path)

    # create all the required subfolders
    os.makedirs(os.path.join(true_path, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(true_path, 'train/masks'), exist_ok=True)
    os.makedirs(os.path.join(true_path, 'val/images'), exist_ok=True)
    os.makedirs(os.path.join(true_path, 'val/masks'), exist_ok=True)

    image_size = args.image_size
    if image_size is not None:
        # size compatible for cv2 resize
        image_size = tuple([image_size[1], image_size[0]])

    # compute all the outputs
    all_outputs = save_outputs(model, directory, backbone, image_size)

    # number of principal components to be extracted
    n_components = 3
    n_select = 1
    cluster(all_outputs, json_path, n_components=n_components)

    read_mapping(true_mapping_path, json_path, true_path, directory=directory, dims_to_take=n_select)


if __name__=="__main__":
    main()