"""
Generate the prompt.json and prompt_part.json files from the Chaoyang file.
"""
import pandas as pd
import json 
import os
import glob
import ipdb
from tqdm import tqdm

# TODO:
# 1. read the train json file of chaoyang dataset. The train json file could be changed.
# 2. Create the prompt.json file and prompt file. --> Be careful: the prompt need to be generated from the mutli-labels.
# 3. The prompt.json file should be saved in the chaoyang folder.


def save_prompt_json(data, prompt_path):
    # just save the data to the prompt_path. The data is a DataFrame.
    data_dict = data.to_dict('records')
    with open(prompt_path, 'w') as f:
        json.dump(data_dict, f, ensure_ascii=False)


# Please modify the below code as the function.
def extract_json(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError("The json file does not exist. Please check the file.")
    else:
        with open(json_file, 'r') as f:
            data = json.load(f)
            data = pd.DataFrame(data)

    # return DataFrame.
    return data

def gene_prompts(data, mode):
    """
    data: DataFrame
    """
    # in the DataFrame, the label_A, label_B, label_C are the multi-labels. Each label has four classes.
    # 4 classes: 0 -> normal, 1 -> serrated, 2 -> adenocarcinoma, 3 -> adenoma.
    # Please help me map the label to the class.
    label_map = {0: 'normal', 1: 'serrated', 2: 'adenocarcinoma', 3: 'adenoma'}
    if mode == 'text':
        template = "A multi-rater picture of histopathology. Three different annotators have labeled this image as {}, {} and {}." 
    elif mode == 'words':
        template = "{}, {}, {}"
    else:
        raise ValueError("The mode is not supported. Please check the mode.")
    for label in ['label_A', 'label_B', 'label_C']:
        data[label] = data[label].apply(lambda x: label_map[x])
    # data['prompt'] = template.format(data['label_A'], data['label_B'], data['label_C'])
    data['prompt'] = data.apply(lambda row: template.format(row['label_A'], row['label_B'], row['label_C']), axis=1)
    data['source'] = data['name'].apply(lambda x: 'cy_edge/' + x.split('/')[-1][:-4] + '.npy')
    # modify the column name of extracted_data.  name -> target
    data.rename(columns={'name': 'target'}, inplace=True)
    data_extract = data[['prompt', 'target', 'source']]
    return data_extract


def gene_test_prompts(data, mode):
    """
    data: DataFrame
    mode: str, 'text' or 'words'
    """
    # in the DataFrame, the label_A, label_B, label_C are the multi-labels. Each label has four classes.
    # 4 classes: 0 -> normal, 1 -> serrated, 2 -> adenocarcinoma, 3 -> adenoma.
    # Please help me map the label to the class.
    label_map = {0: 'normal', 1: 'serrated', 2: 'adenocarcinoma', 3: 'adenoma'}
    if mode == 'text':
        # TODO: Add more text templates for randomly choosing.
        template = "A multi-rater picture of histopathology. Three different annotators have labeled this image as {}, {} and {}." 
    elif mode == 'words':
        template = "{}, {}, {}"
    else:
        raise ValueError("The mode is not supported. Please check the mode.")
    # for label in ['label_A', 'label_B', 'label_C']:
    #     data[label] = data[label].apply(lambda x: label_map[x])
    data['prompt'] = data.apply(lambda row: template.format(label_map[row['label_A']], label_map[row['label_B']], label_map[row['label_C']]), axis=1)
    image_files = [npy for npy in os.listdir('./training/chaoyang/test_edge') if npy.endswith('.npy')] 

    # As I generate all the images of train and test dataset, so to filter those images which are not in the train dataset.
    # assert len(image_files) == 5 * len(data['name'].to_list()), \
    # f"The number of edge files {str(len(image_files))} is not equal to the predicted number of images {str(5*len(data['name']))}."
    new_rows = []
    for name in tqdm(data['name'].to_list()):
        image_prefix = name.split('/')[-1].split('.')[0]
        matching_files = [file for file in image_files if file.startswith(image_prefix)]
        # Add the matching_files into the data.
        for file in matching_files:
            new_row = data[data['name'] == name].copy()  # actually, the source column does not exist in the original data. So 
                                                        # if create a new row, the original data will not be changed. Keep Nan values. 
            assert new_row.shape[0] == 1, "The new_row should have only one row."
            new_row['gen_source'] = file[:-4] + '.jpg'  # TODO: check the generated image name in file of test_chaoyang.py.
            new_row['source'] = os.path.join('test_edge', file)
            new_row['prompt'] = template.format(new_row['label_A'].values[0], new_row['label_B'].values[0], new_row['label_C'].values[0])
            new_rows.append(new_row)
    
    new_data = pd.concat(new_rows, ignore_index=True)

    # save the new_data for chaoyang dataset later processing.
    save_prompt_json(new_data, './training/chaoyang/original_and_gen.json') 
    # # modify the column name of extracted_data.  name -> target
    new_data.rename(columns={'name': 'target'}, inplace=True)
    data_extract = new_data[['prompt', 'target', 'source']]

    assert data_extract['source'].isnull().sum() == 0, "The source column should not have nan values."
    
    return data_extract


if __name__ == '__main__':
    split = 'test'
    json_file = './training/chaoyang/json/train_split_2.json'
    data = extract_json(json_file)
    if split == 'train':
        data_extract = gene_prompts(data, mode='text')
        save_prompt_json(data_extract, './training/chaoyang/prompt.json')
    elif split == 'test':
        test_edge_list = os.listdir('./training/chaoyang/test_edge')
        data_extract = gene_test_prompts(data, mode='text')
        save_prompt_json(data_extract, './training/chaoyang/test_prompt.json')