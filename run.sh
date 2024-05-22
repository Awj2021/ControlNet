#!/bin/bash

set -e 

# python train_quilt1M.py

# After finetune after the quilt1M, continue finetuning the ControlNet with Chaoyang Dataset.

# /sda2/wenjieProject/miniconda3/envs/control_a61/bin/python edge_gen_cy.py
# /sda2/wenjieProject/miniconda3/envs/control_a61/bin/python json_gen_cy.py
# /sda2/wenjieProject/miniconda3/envs/control_a61/bin/python train_chaoyang.py


## Generate the different prompt files for the Chaoyang Dataset.
# python json_gen_cy.py --prompt_mode "words" --split "train"

# export CUDA_VISIBLE_DEVICES=0
# python test_chaoyang.py --prompt_path ./training/chaoyang/test_prompt_part_1.json

# export CUDA_VISIBLE_DEVICES=0
# python test_chaoyang.py --prompt_path /sda2/wenjieProject/ControlNet/training/chaoyang/test_prompt_part_2.json

# export CUDA_VISIBLE_DEVICES=1
# python test_chaoyang.py --prompt_path ./training/chaoyang/test_prompt_part_3.json

# export CUDA_VISIBLE_DEVICES=1
# python test_chaoyang.py --prompt_path ./training/chaoyang/test_prompt_part_4.json


# version of words:
# Use the words prompt to test the Chaoyang Dataset.
export CUDA_VISIBLE_DEVICES=0 
python test_chaoyang.py --prompt_path ./training/chaoyang/version_words/test_prompt_part_1.json

export CUDA_VISIBLE_DEVICES=0 
python test_chaoyang.py --prompt_path ./training/chaoyang/version_words/test_prompt_part_2.json

# python test_chaoyang.py --prompt_path ./training/chaoyang/version_words/test_prompt_part_3.json

# python test_chaoyang.py --prompt_path ./training/chaoyang/version_words/test_prompt_part_4.json