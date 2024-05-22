"""
Generate the prompt.json and prompt_part.json files from the Chaoyang file.
"""

import pandas as pd
import json
import os
import glob
import ipdb
from tqdm import tqdm
import random
import argparse
import yaml


# 1. read the **train json** file of chaoyang dataset. The train json file could be changed.
# 2. Create the prompt.json file and prompt file. --> Be careful: the prompt need to be generated from the mutli-labels.
# 3. The prompt.json file should be saved in the chaoyang folder.

## Pay Attention: the data used for generating the json file is train json. Just used for generating the images, rather
## than classification.


def save_prompt_json(data, prompt_path):
    # just save the data to the prompt_path. The data is a DataFrame.
    data_dict = data.to_dict("records")
    with open(prompt_path, "w") as f:
        json.dump(data_dict, f, ensure_ascii=False)


# Please modify the below code as the function.
def extract_json(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError("The json file does not exist. Please check the file.")
    else:
        with open(json_file, "r") as f:
            data = json.load(f)
            data = pd.DataFrame(data)

    # return DataFrame.
    return data


def split_prompts(data, args):
    """
    data: DataFrame
    """
    print("============  Splitting the prompt file...")
    # with open(args.gen_folder, 'rt') as f:
    # data = json.load(f)
    # convert the data to json
    json_data = data.to_json(orient="records")
    data = json.loads(json_data)

    # split the json file into 4 parts and saved them.
    part_size = len(data) // 4
    parts = [data[i * part_size : (i + 1) * part_size] for i in range(4)]

    # If the data size is not a multiple of 4, add the remaining items to the last part
    if len(data) % 4 != 0:
        parts[-1] += data[4 * part_size :]

    # Save each part separately
    for i, part in enumerate(parts):
        with open(
            os.path.join(config["version"]["folder"], f"test_prompt_part_{i+1}.json"),
            "wt",
        ) as f:
            json.dump(part, f)


def gene_prompts(data, config):
    """
    data: DataFrame
    """
    # in the DataFrame, the label_A, label_B, label_C are the multi-labels. Each label has four classes.
    # 4 classes: 0 -> normal, 1 -> serrated, 2 -> adenocarcinoma, 3 -> adenoma.
    # Please help me map the label to the class.
    label_map = {0: "normal", 1: "serrated", 2: "adenocarcinoma", 3: "adenoma"}
    if config["dataset"]["prompt_mode"] == "text":
        template = "A multi-rater picture of histopathology. Three different annotators have labeled this image as {}, {} and {}."
    elif config["dataset"]["prompt_mode"] == "words":
        template = "{}, {}, {}"
    else:
        raise ValueError("The mode is not supported. Please check the mode.")
    for label in ["label_A", "label_B", "label_C"]:
        data[label] = data[label].apply(lambda x: label_map[x])
    # data['prompt'] = template.format(data['label_A'], data['label_B'], data['label_C'])
    # data["prompt"] = data.apply(
    #     lambda row: template.format(row["label_A"], row["label_B"], row["label_C"]),
    #     axis=1,
    # )
    data["prompt"] = data.apply(
        lambda row: template.format(row["label_A"], row["label_B"], row["label_C"]),
        axis=1,
    )
    data["source"] = data["name"].apply(
        lambda x: "cy_edge/" + x.split("/")[-1][:-4] + ".npy"
    )
    # modify the column name of extracted_data.  name -> target
    data.rename(columns={"name": "target"}, inplace=True)
    data_extract = data[["prompt", "target", "source"]]
    return data_extract


def gene_test_prompts(data, config):
    """
    data: DataFrame.
    mode: str, 'text' or 'words'
    """
    # in the DataFrame, the label_A, label_B, label_C are the multi-labels. Each label has four classes.
    # 4 classes: 0 -> normal, 1 -> serrated, 2 -> adenocarcinoma, 3 -> adenoma.
    # Please help me map the label to the class.
    label_map = {0: "normal", 1: "serrated", 2: "adenocarcinoma", 3: "adenoma"}

    # First Step: Load the data, and do the statistics of the labels combinations.
    # ipdb.set_trace()
    if config["dataset"]["prompt_mode"] == "text":
        # TODO: Add more text templates for randomly choosing.
        template = "A multi-rater picture of histopathology. Three different annotators have labeled this image as {}, {} and {}."
    elif config["dataset"]["prompt_mode"] == "words":
        template = "{}, {}, {}"
    else:
        raise ValueError("The mode is not supported. Please check the mode.")

    for label in ["label_A", "label_B", "label_C"]:
        data[label] = data[label].apply(lambda x: label_map[x])

    data["prompt"] = data.apply(
        lambda row: template.format(row["label_A"], row["label_B"], row["label_C"]),
        axis=1,
    )
    # As I generate all the images of train and test dataset, so to filter those images which are not in the train dataset.
    # assert len(image_files) == 5 * len(data['name'].to_list()), \
    # f"The number of edge files {str(len(image_files))} is not equal to the predicted number of images {str(5*len(data['name']))}."

    # # # version0: matching the edge maps with the original images.
    new_rows = []
    image_files = [
        npy
        for npy in os.listdir(config["dataset"]["edge_test_path"])
        if npy.endswith(".npy")
    ]
    for name in tqdm(data["name"].to_list()):
        image_prefix = name.split("/")[-1].split(".")[0]
        matching_files = [file for file in image_files if file.startswith(image_prefix)]
        # Add the matching_files into the data.
        for file in matching_files:
            # ipdb.set_trace()
            new_row = data[
                data["name"] == name
            ].copy()  # actually, the source column does not exist in the original data. So
            # if create a new row, the original data will not be changed. Keep Nan values.
            assert new_row.shape[0] == 1, "The new_row should have only one row."
            new_row["source"] = os.path.join(
                os.path.basename(config["dataset"]["edge_test_path"]), file
            )

            # The generated images directory.
            new_row["name"] = os.path.join(
                os.path.basename(config["dataset"]["gen_folder"]),
                file[:-4] + ".jpg",
            )
            new_row["prompt"] = template.format(
                new_row["label_A"].values[0],
                new_row["label_B"].values[0],
                new_row["label_C"].values[0],
            )
            new_rows.append(new_row)

    new_data = pd.concat(new_rows, ignore_index=True)

    # # # version1: randomly choose the edge maps.
    # new_rows = []
    # edge_map_names = data['name'].apply(lambda x: x.split('/')[-1][:-4]).to_list()
    # # Actually the edge_maps include the train and test edge maps.
    # # the npy is: 537688_1-IMG005x003-3_70_190.npy
    # # element_map_names: 537688_1-IMG005x003-3

    # edge_maps = [npy for npy in os.listdir(args.test_edge_folder) if npy.endswith('.npy') and '_'.join(npy.split('_')[:-2]) in edge_map_names]
    # # ipdb.set_trace()
    # assert len(edge_maps) == 2 * len(data['name'].to_list()), \
    #     f"The number of edge maps {len(edge_maps)} is not equal to the 2 times number of original images {2 * len(data['name'].to_list())}."
    # random.shuffle(edge_maps)  # Shuffle the edge_maps randomly
    # for name in tqdm(data['name'].to_list()):
    #     for i in range(2):
    #         edge_map = edge_maps.pop()
    #         new_row = data[data['name'] == name].copy()
    #         assert new_row.shape[0] == 1, "The new_row should have only one row."
    #         new_row['name'] = os.path.join('gen_test_edge', edge_map[:-4] + '.jpg')
    #         new_row['source'] = os.path.join('test_edge', edge_map)
    #         # new_row['prompt'] = template.format(new_row['label_A'].values[0], new_row['label_B'].values[0], new_row['label_C'].values[0])
    #         new_rows.append(new_row)
    # new_data = pd.concat(new_rows, ignore_index=True)

    original_and_gen = pd.concat([data, new_data], ignore_index=True)
    original_and_gen = original_and_gen[
        ["label", "label_A", "label_B", "label_C", "name", "prompt"]
    ]

    #### The data_mix.json is used for further classification.
    save_prompt_json(
        original_and_gen, os.path.join(config["version"]["folder"], "data_mix.json")
    )  # save it for further classification.
    # # modify the column name of extracted_data.  name -> target
    new_data.rename(columns={"name": "target"}, inplace=True)
    data_extract = new_data[["prompt", "target", "source"]]
    split_prompts(data_extract, args)

    assert (
        data_extract["source"].isnull().sum() == 0
    ), "The source column should not have nan values."

    return data_extract


def argparser():
    parser = argparse.ArgumentParser(
        description="Test the model on the Chaoyang test dataset."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "all"],
        help="The split of the dataset. train or test.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()

    with open("config_cy_split_1_words.yaml", "r") as f:
        config = yaml.safe_load(f)
    split = args.split
    json_file = config["train_chaoyang"][
        "train_json"
    ]  # The original json file of the Chaoyang dataset.
    if not os.path.exists(
        os.path.join(config["version"]["folder"], config["dataset"]["gen_folder"])
    ):
        os.makedirs(
            os.path.join(config["version"]["folder"], config["dataset"]["gen_folder"])
        )

    data_wo_copy = extract_json(json_file)  # DataFrame.
    if split == "train":
        data_extract = gene_prompts(data_wo_copy.copy(), config=config)
        save_prompt_json(
            data_extract,
            os.path.join(
                config["version"]["folder"], config["dataset"]["prompt_train_name"]
            ),
        )
    elif split == "test":
        data_extract = gene_test_prompts(data_wo_copy.copy(), config=config)
        save_prompt_json(
            data_extract,
            os.path.join(
                config["version"]["folder"], config["dataset"]["prompt_test_name"]
            ),
        )

    elif split == "all":
        data_extract = gene_prompts(data_wo_copy.copy(), config=config)
        save_prompt_json(
            data_extract,
            os.path.join(
                config["version"]["folder"], config["dataset"]["prompt_train_name"]
            ),
        )

        data_extract_test = gene_test_prompts(data_wo_copy.copy(), config=config)
        save_prompt_json(
            data_extract_test,
            os.path.join(
                config["version"]["folder"], config["dataset"]["prompt_test_name"]
            ),
        )

    else:
        raise ValueError("The split is not supported. Please check the split.")
