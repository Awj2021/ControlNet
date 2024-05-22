## Finetune Steps
1. Write and rename the config files. E.g., config_cy_ori_split_text.yaml
2. Construct the training and testing dataset.
3. Getting the json files.
4. Finetune the Chaoyang dataset.
5. Convert the pretrained model into diffusers' format.
6. Inference and get the data-augmented dataset.
7. Train the classification models.

## Details
1. Write and rename the config files
Please keep the same suffix and prefix in the file. You will see, there will be several items that you'd to change. Please keep your eyes on it!


2. Construct the training and testing dataset.
If you want to change the train/test split, or the data-augmented scheduler,
you need to go through this step. For the generated edge maps, please save at different folders. You also need to change and choose the which json file.

`python edge_gen_cy.py`

3. For continue finetuning, to load the train / test dataset, you'd better to use the json file to load them. For example, generating the .json files and save in a specific folder. You also need to change and choose the json file.

`python json_gen_cy.py`

4. Finetune the Chaoyang dataset.
Before running the file, please check the loaded config file in the end of file.  
`python train_chaoyang.py` 

5. Convert the pretrained model into diffusers' format.
Changing the training environment, e.g., `test`;  
Going into the directory of `/sda2/wenjieProject/diffusers`;
Running the command: 
`python ./scripts/convert_controlnet_to_diffusers.py \
    --checkpoint_path /sda2/wenjieProject/ControlNet/logs/ft_cy_train_cy_ori_split_text_ulws070.surrey.ac.uk/models_text/control_cy_ori_split_99_text.ckpt \
    --dump_path logs/ft_cy_train_cy_ori_split_text_ulws070.surrey.ac.uk/models_text/epoch_99`


6. Inference and get the data-augmented dataset.
Running the file:
`python test_chaoyang.py --prompt_path xxx --config_path xxx`

7. Train the classification models. (Maybe directly running on condor.)

