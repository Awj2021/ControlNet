# Description: Test the model on the test dataset.
# Attention: This script is tested on the environment of the 'test'. For more information, please refer to: https://mp.csdn.net/mp_blog/creation/editor/138912152

####################### below is the test code for one image. #######################
# from diffusers import StableDiffusionControlNetPipeline
# from diffusers.utils import load_image


# canny_image = load_image('./training/chaoyang/test_edge/538880-3-IMG013x018-3_90_130.jpg')
# canny_text = 'A multi-rater picture of histopathology. Three different annotators have labeled this image as serrated, adenocarcinoma and adenocarcinoma.'

# pipe = StableDiffusionControlNetPipeline.from_pretrained("./logs/models/finetuning_chaoyang_ulws070.surrey.ac.uk/control_cy").to("cuda")
# pipe.safety_checker = lambda images, clip_input: (images, False)

# image = pipe(prompt=canny_text, controlnet_hint=canny_image).images[0]
# image.save("generated.png")

# based on the test code, load the test dataset and then run the model on the test dataset.

####################### above is the test code for one image. #######################


####################### below is the test code running under the enironment test. #######################
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from finetune_dataset import ChaoyangTestDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import ipdb
import json
import os
from itertools import islice
from accelerate import Accelerator
import argparse
import logging
import yaml


# move the split_prompts function to the json_gen_cy.py


def argparser():
    parser = argparse.ArgumentParser(
        description="Test the model on the Chaoyang test dataset."
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="./training/version_split_2_words/test_prompt.json",
        help="The path of the prompt file.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="./training/chaoyang/version1/test_prompt.json",
        help="The path of the prompt file.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # load the test dataset
    args = argparser()
    logger_name = os.path.basename(args.prompt_path).split(".")[0]
    logging.basicConfig(filename=f"{logger_name}.log", level=logging.INFO)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # TODO: if you need to split the prompt file, please uncomment the below code.
    # split_prompts(prompt_path='./training/chaoyang/test_prompt.json')
    # ipdb.set_trace()
    accelerator = Accelerator()
    dataset = ChaoyangTestDataset(prompt_path=args.prompt_path)

    # You cannot use batch_size > 1.
    # As when you generate the images, you cannot get a batch of images.
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)
    dataloader = accelerator.prepare(dataloader)

    # load the model
    logging.info("Loading the model...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config["test_chaoyang"]["model_dir"]
    )
    pipe = pipe.to(accelerator.device)

    pipe.safety_checker = lambda images, clip_input: (images, False)
    logging.info("Model is loaded.")
    # run the model on the test dataset
    gen_dir = config["test_chaoyang"]["gen_dir"]
    # generator = torch.Generator("cuda").manual_seed(14556)
    if not os.path.exists(gen_dir):
        logging.info(f"Creating the directory: {gen_dir}")
        os.makedirs(gen_dir)

    print("Start generating the images...")
    ################## Tooooooooo sloooooooow ##################
    # for _, data in islice(tqdm(enumerate(dataloader)), 10):
    for i, data in tqdm(enumerate(dataloader)):
        logging.info(f"Generating the {i}-th image...")
        prompt = data["txt"][0]
        controlnet_hint = data["hint"][0]
        name = data["name"][0]
        image = pipe(
            prompt=prompt, controlnet_hint=controlnet_hint, num_inference_steps=40
        ).images[0]
        save_name = os.path.join(gen_dir, name[:-4] + ".jpg")
        image.save(save_name)

    # for i, data in islice(tqdm(enumerate(dataloader)), 2):
    #     logging.info(f"Generating the {i}-th image...")
    #     prompts = data['txt']
    #     controlnet_hints = data['hint']
    #     names = data['name']
    #     ipdb.set_trace()
    #     images = pipe(prompt=prompts, controlnet_hint=controlnet_hints, num_inference_steps=40).images[0]
    #     # save_name = os.path.join(gen_dir, name[:-4] + '.jpg')
    #     # image.save(save_name)
