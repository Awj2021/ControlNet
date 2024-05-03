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

def Canny_images(img_list, data_dir, result_dir, image_solution, low_threshold, high_threshold):
    """
    img_list: image list, especially the names of images.
    data_dir: data folder of Dataset.
    image_solution: the size after resized.
    Low_threshold: used for canny algorithm, please check the value.
    high_threshold: used for canny algorithm.
    """
    with Pool() as p:
        p.map(process_image, img_list)

    print("Finishing Processing!!!")


if __name__ == '__main__':
    # Firstly judge the image list file exists or not.
    # Only save the part of the image list. 

    ################ Attention: Please Change the switch before running. ################
    test_part = False
    if test_part:
        if not os.path.exists('./training/quilt_1M_img_list_part.txt'):
            print("Please check the image list file. Now generating is starting...")
            df = pd.read_csv('./training/quilt_1M_lookup.csv')
            # filter the duplicated images.
            print(f"Before filtering: {len(df)}")
            df = df.drop_duplicates(subset='image_path')
            img_list = df['image_path'].to_list()
            print(f"Total images: {len(img_list)}")
            img_list = img_list[:1000]  # with the suffix of .jpg
        # save the img_list for further use.
            with open('./training/quilt_1M_img_list_part.txt', 'w') as f:
                for img in img_list:
                    f.write(img + '\n')
        else:
            with open('./training/quilt_1M_img_list_part.txt', 'r') as f:
                img_list = f.readlines()
                img_list = [img.strip() for img in img_list]
    else:
        if not os.path.exists('./training/quilt_1M_img_list.txt'):
            print("Please check the image list file. Now generating is starting...")
            df = pd.read_csv('./training/quilt_1M_lookup.csv')
            # filter the duplicated images.
            print(f"Before filtering: {len(df)}")
            df = df.drop_duplicates(subset='image_path')
            img_list = df['image_path'].to_list()
            print(f"Total images: {len(img_list)}")

        # save the img_list for further use.
            with open('./training/quilt_1M_img_list.txt', 'w') as f:
                for img in img_list:
                    f.write(img + '\n')
        else:
            with open('./training/quilt_1M_img_list.txt', 'r') as f:
                img_list = f.readlines()
                img_list = [img.strip() for img in img_list]

    data_dir = './training/quilt_1m'
    result_dir = './training/quilt_1m_edge_part' if test_part else './training/quilt_1m_edge'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    image_solution = 512
    low_threshold =  50
    high_threshold =  150
    apply_canny = CannyDetector()
    # Attention: the original img_list has several duplicated images. So please filter the duplicated images firstly. 
    Canny_images(img_list, data_dir, result_dir, image_solution, low_threshold, high_threshold)