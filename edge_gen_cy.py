"""
Generate the edge map for chaoyang dataset. 
"""

import sys

sys.path.insert(0, "./ControlNet")
from share import *
import cv2
import einops
import numpy as np
import pandas as pd
import os
import ipdb

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import itertools
import random
from skimage import io, transform, img_as_ubyte
from tqdm import tqdm
import yaml


def process_train_image(
    img_name, low_threshold, high_threshold, image_solution, data_dir, result_dir
):
    try:
        img = cv2.imread(os.path.join(data_dir, img_name))
        img = resize_image(HWC3(img), image_solution)
        H, W, C = img.shape
        detected_map = apply_canny(HWC3(img), low_threshold, high_threshold)

        np.save(os.path.join(result_dir, f"{img_name[:-4]}.npy"), detected_map)
        cv2.imwrite(os.path.join(result_dir, f"{img_name}.jpg"), detected_map)
    except Exception as e:
        print(f"Error in processing {img_name}: {e}")


def Canny_train_images(result_dir):
    """
    result_dir: the directory of the result images.
    Function: generate the edge maps for the training and testing images. During the training,
            we only load the training images.
    """
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for split in ["train", "test"]:
        data_dir = f"./training/chaoyang/{split}"
        img_list = os.listdir(data_dir)

    image_solution = 512
    low_threshold = 50
    high_threshold = 150
    img_args_list = [
        (img_name, low_threshold, high_threshold, image_solution, result_dir)
        for img_name in img_list
    ]
    with Pool() as p:
        p.starmap(process_train_image, img_args_list)

    print("Finishing Processing!!!")


def process_test_image(
    image, low_threshold, high_threshold, image_solution, result_dir
):
    try:
        img = cv2.imread(image)
        # TODO: add the image crop and rotation.
        # randomly image rotation.
        img_rotate = img_as_ubyte(
            transform.rotate(img, random.choice([0, 90, 180, 270]))
        )
        # randomly image crop.
        crop_h, crop_w = 430, 430  # keep almost the same size as the original image.
        start_y = np.random.randint(0, img.shape[0] - crop_h)
        start_x = np.random.randint(0, img.shape[1] - crop_w)

        img_rorate_crop = img_rotate[
            start_y : start_y + crop_h, start_x : start_x + crop_w, :
        ]
        img_rorate_crop = resize_image(HWC3(img_rorate_crop), image_solution)
        H, W, C = img_rorate_crop.shape

        detected_map = apply_canny(HWC3(img_rorate_crop), low_threshold, high_threshold)

        img_name = image.split("/")[-1]
        result_name = f"{img_name[:-4]}_{low_threshold}_{high_threshold}.npy"
        np.save(os.path.join(result_dir, result_name), detected_map)
        cv2.imwrite(os.path.join(result_dir, f"{result_name[:-4]}.jpg"), detected_map)
    except Exception as e:
        print(f"Error in processing {image}: {e}")
        raise Exception(e)


def Canny_test_images(result_dir):
    """
    result_dir: the directory of the result images.
    Function: generate the edge maps for the training and testing images. Actually,
                Generate all the images in the Chaoyang Dataset.
    """
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    img_list = []
    for split in ["train", "test"]:
        data_dir = f"./training/chaoyang/{split}"
        img_list.extend([os.path.join(data_dir, f) for f in os.listdir(data_dir)])
    print(f"Total images: {len(img_list)}")

    image_solution = 512
    low_thresholds = [30, 70, 110]
    high_thresholds = [130, 170, 210]

    # Firstly Test some images.
    low_high_lists = list(itertools.product(low_thresholds, high_thresholds))
    for image in tqdm(img_list):
        chosen_pairs = random.sample(
            low_high_lists, 2
        )  # randomly choose two pairs for generation.
        for low_high in chosen_pairs:
            low_threshold, high_threshold = low_high
            process_test_image(
                image, low_threshold, high_threshold, image_solution, result_dir
            )

    print("Finishing Processing!!!")


if __name__ == "__main__":
    # Due to the split format of chaoyang dataset, we need to generate all the edge maps.e.g., train and test.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    mode = "test"
    apply_canny = CannyDetector()
    if mode == "train":
        result_dir = config["dataset"]["edge_train_path"]
        Canny_train_images(result_dir)
    else:
        result_dir = config["dataset"]["edge_test_path"]
        Canny_test_images(result_dir)
