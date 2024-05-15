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
from tqdm import tqdm
import ipdb
import json
import os
from itertools import islice


if __name__ == '__main__':
    # load the test dataset
    dataset = ChaoyangTestDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)
    # load the model
    print("Loading the model...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained("./logs/models/finetuning_chaoyang_ulws070.surrey.ac.uk/control_cy").to("cuda")
    pipe.safety_checker = lambda images, clip_input: (images, False)
    print("Model is loaded.")
    # run the model on the test dataset
    gen_dir = './training/chaoyang/gen_test_edge'
    if not os.path.exists(gen_dir):
        print(f"Creating the directory: {gen_dir}")
        os.makedirs(gen_dir)

    print("Start generating the images...")   

    # for _, data in islice(tqdm(enumerate(dataloader)), 2):
    for _, data in tqdm(enumerate(dataloader)):
        prompt = data['txt'][0]
        controlnet_hint = data['hint'][0]
        name = data['name'][0]
        image = pipe(prompt=prompt, controlnet_hint=controlnet_hint).images[0]
        save_name = os.path.join(gen_dir, name[:-4] + '.jpg')
        image.save(save_name)