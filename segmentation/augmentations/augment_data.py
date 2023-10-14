from ast import arg
import cv2
import numpy as np
import glob
import os
from random import randint
import random
import math
from shapely.geometry import Polygon, Point
from statistics import mean
import argparse

kernel_3 = np.ones((3,3), np.uint8)
kernel_5 = np.ones((5,5), np.uint8)
kernel_8 = np.ones((8,8), np.uint8)

def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds

    points = []
    while len(points) < num_points:
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if (random_point.within(poly)):
            points.append(random_point)
    
    return points

def check_start_end(pt, side, minus):
    if minus:
        if pt - side < 0:
            return pt
        return (pt - side)
    else:
        if pt + side > 448:
            return pt
        return (pt + side)

def get_side_range(c, mask):
    init_ls = np.zeros(mask.shape, dtype=np.int8)
    approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
    cv2.drawContours(init_ls, [approx], 0, 255, thickness=cv2.FILLED)
    init_ls = cv2.threshold(np.uint8(init_ls), int(mean(np.unique(init_ls)))/2, 255, cv2.THRESH_BINARY)[1]
    dist = cv2.distanceTransform(init_ls, cv2.DIST_L1, 3)
    width = 2*dist.max() - 1
    if dist.max() > 0.5:
        width = 2*dist.max() - 1
        s = width - width // 4
        e = width + width // 4
        if width - width // 2 < 5:
            s = 5
            e = 5 + 5 // 4
        elif width - width//2 > 20:
            s = 35 - 35 // 4
            e = 35
    else:
        s = 2
        e = 2
    print(s, e)
    return (s,e)

def random_masking(mask):
    thresh = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        if cv2.contourArea(c) <= 50:
            continue
            
        s, e = get_side_range(c, mask)

        contour = np.squeeze(c)
        polygon = Polygon(contour)

        if cv2.contourArea(c) <= 100:
            n = randint(1, 3)
        elif cv2.contourArea(c) <= 200:
            n = randint(2, 5)
        else:
            n = randint(5, 7)
        
        print(n)
        points = random_points_within(polygon, n)

        for p in points:
            print(p.x, ",", p.y)
            x = math.floor(p.x)
            y = math.floor(p.y)
            custom_square_side = randint(s, e)
            mask[check_start_end(x, custom_square_side, minus=True):check_start_end(x, custom_square_side, minus=False),
                    check_start_end(y, custom_square_side, minus=True):check_start_end(y, custom_square_side, minus=False)]
    return mask

def augment_images(orig_image_path, orig_mask_path, new_image_path, new_mask_path, mdil, rmask, spline=False, mix=False):

    for directory_path in glob.glob(orig_image_path):
        for img_path in sorted(glob.glob(os.path.join(directory_path, "*.jpg"))):
            img = cv2.imread(img_path, 1)
            
            cv2.imwrite(os.path.join(new_image_path, img_path.split("/")[-1]), img)

            if mdil:
                print(img_path)
                if mix:
                    print("mdil operations mix")
                    cv2.imwrite(os.path.join(new_image_path, img_path.split("/")[-1].split(".")[0] + "_dilated_3.jpg"), img)
                    cv2.imwrite(os.path.join(new_image_path, img_path.split("/")[-1].split(".")[0] + "_dilated_5.jpg"), img)
                else:
                    print("mdil operations")
                    cv2.imwrite(os.path.join(new_image_path, img_path.split("/")[-1].split(".")[0] + "_dilated_3.jpg"), img)
                    cv2.imwrite(os.path.join(new_image_path, img_path.split("/")[-1].split(".")[0] + "_dilated_5.jpg"), img)
                    cv2.imwrite(os.path.join(new_image_path, img_path.split("/")[-1].split(".")[0] + "_dilated_8.jpg"), img)
            
            if rmask:
                if mix:
                    print('rmask operations mix')
                else:
                    print('rmask operations')
                print(img_path)
                cv2.imwrite(os.path.join(new_image_path, img_path.split("/")[-1].split(".")[0] + "_masked.jpg"), img)

        for directory_path in glob.glob(orig_mask_path):
            for mask_path in sorted(glob.glob(os.path.join(directory_path, '*.png'))):
                mask = cv2.imread(mask_path, 0)
                cv2.imwrite(os.path.join(new_mask_path, mask_path.split("/")[-1].split(".")[0] + ".png"), mask)

                if mdil:
                    print(mask_path)
                    mask_copy = mask.copy()
                    SIZE_X, SIZE_Y = mask.shape
                    if spline:
                        print("mdil operations scale space")
                        mask_copy = cv2.resize(mask_copy, (SIZE_Y*4, SIZE_X*4), interpolation=cv2.INTER_CUBIC)
                        if mix:
                            mask_3 = cv2.dilate(mask_copy, kernel_3, iterations=1)
                            mask_3 = cv2.resize(mask_3, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
                            mask_3 = cv2.threshold(mask_3, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                            cv2.imwrite(os.path.join(new_mask_path, mask_path.split("/")[-1].split(".")[0] + "_dilated_3.png"), mask_3)

                            mask_5 = cv2.dilate(mask_copy, kernel_5, iterations=1)
                            mask_5 = cv2.resize(mask_5, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
                            mask_5 = cv2.threshold(mask_5, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                            cv2.imwrite(os.path.join(new_mask_path, mask_path.split("/")[-1].split(".")[0] + "_dilated_5.png"), mask_5)

                        else:
                            print('mdil operations scale space mdil')

                            mask_3 = cv2.dilate(mask_copy, kernel_3, iterations=1)
                            mask_3 = cv2.resize(mask_3, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
                            mask_3 = cv2.threshold(mask_3, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                            cv2.imwrite(os.path.join(new_mask_path, mask_path.split("/")[-1].split(".")[0] + "_dilated_3.png"), mask_3)

                            mask_5 = cv2.dilate(mask_copy, kernel_5, iterations=1)
                            mask_5 = cv2.resize(mask_5, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
                            mask_5 = cv2.threshold(mask_5, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                            cv2.imwrite(os.path.join(new_mask_path, mask_path.split("/")[-1].split(".")[0] + "_dilated_5.png"), mask_5)

                            mask_8 = cv2.dilate(mask_copy, kernel_8, iterations=1)
                            mask_8 = cv2.resize(mask_8, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
                            mask_8 = cv2.threshold(mask_8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                            cv2.imwrite(os.path.join(new_mask_path, mask_path.split("/")[-1].split(".")[0] + "_dilated_8.png"), mask_8)
                    else:
                        if mix:
                            print('mdil operations mix')
                            mask_3 = cv2.dilate(mask_copy, kernel_3, iterations=1)
                            cv2.imwrite(os.path.join(new_mask_path, mask_path.split("/")[-1].split(".")[0] + "_dilated_3.png"), mask_3)

                            mask_5 = cv2.dilate(mask_copy, kernel_5, iterations=1)
                            cv2.imwrite(os.path.join(new_mask_path, mask_path.split("/")[-1].split(".")[0] + "_dilated_5.png"), mask_5)

                        else:
                            print('mdil operations')
                            mask_3 = cv2.dilate(mask_copy, kernel_3, iterations=1)
                            cv2.imwrite(os.path.join(new_mask_path, mask_path.split("/")[-1].split(".")[0] + "_dilated_3.png"), mask_3)

                            mask_5 = cv2.dilate(mask_copy, kernel_5, iterations=1)
                            cv2.imwrite(os.path.join(new_mask_path, mask_path.split("/")[-1].split(".")[0] + "_dilated_5.png"), mask_5)

                            mask_8 = cv2.dilate(mask_copy, kernel_8, iterations=1)
                            cv2.imwrite(os.path.join(new_mask_path, mask_path.split("/")[-1].split(".")[0] + "_dilated_8.png"), mask_8)
                    
                if rmask:
                    if mix:
                        print('rmask operations mix')
                    else:
                        print('rmask operations')
                    print(mask_path)
                    cv2.imwrite(os.path.join(new_mask_path, mask_path.split("/")[-1].split(".")[0] + "_masked.png"), random_masking(mask))


def clean_files(path):
    files = glob.glob(path + "/*")
    for f in files:
        os.remove(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--augmentation', default=0, help="""0 for stochastic width, 1 for stochastic length
                    2 for stochastic scale and 3 for mix augmentation""")
    parser.add_argument('--input_path', default='../training/data/train/')
    parser.add_argument('--output_path', default='augmented/train/')
    args = parser.parse_args()

    original_image_path = os.path.join(args.input_path, "images")
    original_mask_path = os.path.join(args.input_path, "masks")

    output_image_path = os.path.join(args.output_path, "images")
    output_mask_path = os.path.join(args.output_path, "masks")
    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(output_mask_path, exist_ok=True)

    if int(args.augmentation)== 0:
        augment_images(orig_image_path=original_image_path, orig_mask_path=original_mask_path,
                    new_image_path=output_image_path, new_mask_path=output_mask_path,
                    mdil=True, rmask=False)
    elif int(args.augmentation)== 1:
        augment_images(orig_image_path=original_image_path, orig_mask_path=original_mask_path,
                    new_image_path=output_image_path, new_mask_path=output_mask_path,
                    mdil=False, rmask=True)
    elif int(args.augmentation)== 2:
        augment_images(orig_image_path=original_image_path, orig_mask_path=original_mask_path,
                    new_image_path=output_image_path, new_mask_path=output_mask_path,
                    mdil=True, rmask=False, spline=True)
    elif int(args.augmentation)== 3:
        augment_images(orig_image_path=original_image_path, orig_mask_path=original_mask_path,
                    new_image_path=output_image_path, new_mask_path=output_mask_path,
                    mdil=True, rmask=True, mix=True)

if __name__=="__main__":
    main()