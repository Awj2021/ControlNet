import pandas as pd
import json 
import os


def save_prompt_json(folder, data, prompt_path):
    # folder = 'quilt_1m_edge/'
    data['edge_path'] = folder + data['image_path'] 
    data['image_path'] = 'quilt_1m/' + data['image_path']

    # Extract three columns
    extracted_data = data[['caption', 'image_path', 'edge_path']]

    # modify the column names of extracted_data. caption -> prompt, image_path -> target, edge_path -> source
    extracted_data.columns = ['prompt', 'target', 'source']
    data_dict = extracted_data.to_dict('records')
    # Write the list of dictionaries to a JSON file
    with open(prompt_path, 'w') as f:
        json.dump(data_dict, f, ensure_ascii=False)
        

# Please modify the below code as the function.
def extract_json(part_img_list, prompt_path, prompt_part_path):
    data = pd.read_csv('./training/quilt_1M_lookup.csv')

    if part_img_list is not None:
        part_img_list = [img[:-4] + '.jpg' for img in part_img_list]
        data_part = data[data['image_path'].isin(part_img_list)]
        if not os.path.exists(prompt_part_path):
            print("Generating the prompt_part.json file...")
            save_prompt_json('quilt_1m_edge_part/', data_part, prompt_part_path)
        else:
            print(f"{prompt_part_path} File exists. Please check the file.")

    if not os.path.exists(prompt_path):
        print("Generating the prompt.json file...")
        save_prompt_json('quilt_1m_edge/', data, prompt_path)
    else:
        print(f"{prompt_path} File exists. Please check the file.")

if __name__ == '__main__':
    part_img_txt = './training/quilt_1M_img_list_part.txt'
    # read the part_img_txt file and return the list of image names.
    part_img_list = None
    with open(part_img_txt, 'r') as f:
        part_img_list = f.readlines()
        part_img_list = [img.strip() for img in part_img_list]
    extract_json(part_img_list,  './training/quilt_1M_prompt.json', './training/quilt_1M_prompt_part.json')