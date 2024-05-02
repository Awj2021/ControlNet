import pandas as pd
import json 


# Read the file and add a new column
data = pd.read_csv('./training/quilt_1M_lookup.csv')
# The new column is the images names as the column 'image_path'. But I would like to add the folder path before the image name.
data['edge_path'] = 'quilt_1m_edge/' + data['image_path']

# Modify column values
data['image_path'] = 'quilt_1m/' + data['image_path']

# Extract three columns
extracted_data = data[['caption', 'image_path', 'edge_path']]

# modify the column names of extracted_data. caption -> prompt, image_path -> target, edge_path -> source
extracted_data.columns = ['prompt', 'target', 'source']
# Save the extracted data as a JSON file
# extracted_data.to_json('./training/prompt.json', orient='records', lines=True)
# data_dict = extracted_data.to_dict('records')
data_dict = extracted_data.to_dict('records')

# Write the list of dictionaries to a JSON file
with open('./training/prompt.json', 'w') as f:
    json.dump(data_dict, f, ensure_ascii=False)
