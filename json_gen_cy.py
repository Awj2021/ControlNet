"""
Generate the prompt.json and prompt_part.json files from the Chaoyang file.
"""
import pandas as pd
import json 
import os

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

def gene_prompts(data):
    """
    data: DataFrame
    """
    # in the DataFrame, the label_A, label_B, label_C are the multi-labels. Each label has four classes.
    # 4 classes: 0 -> normal, 1 -> serrated, 2 -> adenocarcinoma, 3 -> adenoma.
    # Please help me map the label to the class.
    label_map = {0: 'normal', 1: 'serrated', 2: 'adenocarcinoma', 3: 'adenoma'}
    template = "A multi-rater picture of histopathology. Three different annotators have labeled this image as {}, {} and {}." 
    for label in ['label_A', 'label_B', 'label_C']:
        data[label] = data[label].apply(lambda x: label_map[x])
    # data['prompt'] = template.format(data['label_A'], data['label_B'], data['label_C'])
    data['prompt'] = data.apply(lambda row: template.format(row['label_A'], row['label_B'], row['label_C']), axis=1)
    data['source'] = data['name'].apply(lambda x: 'cy_edge/' + x.split('/')[-1][:-4] + '.npy')
    # modify the column name of extracted_data.  name -> target
    data.rename(columns={'name': 'target'}, inplace=True)
    data_extract = data[['prompt', 'target', 'source']]
    return data_extract


if __name__ == '__main__':
    # part_img_txt = './training/quilt_1M_img_list_part.txt'
    # # read the part_img_txt file and return the list of image names.
    # part_img_list = None
    # with open(part_img_txt, 'r') as f:
    #     part_img_list = f.readlines()
    #     part_img_list = [img.strip() for img in part_img_list]
    # extract_json(part_img_list,  './training/quilt_1M_prompt.json', './training/quilt_1M_prompt_part.json')

    json_file = './training/chaoyang/json/train_split_2.json'
    data = extract_json(json_file)
    data_extract = gene_prompts(data)
    save_prompt_json(data_extract, './training/chaoyang/prompt.json')