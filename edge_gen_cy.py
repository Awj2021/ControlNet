"""
Generate the edge map for chaoyang dataset. 
"""
import sys
sys.path.insert(0, '/vol/research/wenjieProject/projects/owns/ControlNet')
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


def process_image(img_name):
    try:
        img = cv2.imread(os.path.join(data_dir, img_name))
        img = resize_image(HWC3(img), image_solution)
        H, W, C = img.shape

        detected_map = apply_canny(HWC3(img), low_threshold, high_threshold)

        np.save(os.path.join(result_dir, f'{img_name[:-4]}.npy'), detected_map)
        # Due to the size of the image is too large, so we save the image as .npy file.
        # detected_map = HWC3(detected_map)
        # cv2.imwrite(os.path.join(result_dir, f'{img_name}.jpg'), detected_map)
    except Exception as e:
        print(f"Error in processing {img_name}: {e}")

def Canny_images(img_list):
    """
    img_list: image list, especially the names of images.
    """
    with Pool() as p:
        p.map(process_image, img_list)

    print("Finishing Processing!!!")


if __name__ == '__main__':
    # Firstly judge the image list file exists or not.
    # Only save the part of the image list. 
    # For the Chaoyang Dataset, we directly generate all the edge maps.
    ################ Attention: Please Change the switch before running. ################
    # Only for the Chaoyang Train Dataset.
    # Due to the split format of chaoyang dataset, we need to generate all the edge maps.e.g., train and test.
    data_dir = './training/chaoyaong'
    result_dir = './training/chaoyang/cy_edge' 
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    image_solution = 512
    low_threshold =  50
    high_threshold =  150
    apply_canny = CannyDetector()
    for split in ['train', 'test']:
        data_dir = f'./training/chaoyang/{split}'
        img_list = os.listdir(data_dir)

        Canny_images(img_list)
        print(f"Finishing {split} Processing!!!")