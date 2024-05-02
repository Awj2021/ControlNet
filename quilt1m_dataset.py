import json
import cv2
import numpy as np
import ipdb
import os
from torch.utils.data import Dataset


# TODO: add more data augmentation for source images.
class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/prompt.json', 'rt') as f:
            # for line in f:
                # self.data.append(json.loads(line))
            # with open('./training/prompt.json', 'rt') as f:
            self.data = json.load(f)
        self.data_dir = '/vol/research/wenjieProject/projects/owns/ControlNet/training'
        # ipdb.set_trace()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = os.path.join(self.data_dir, item['source'])
        target_filename = os.path.join(self.data_dir, item['target'])
        # ipdb.set_trace()
        assert os.path.exists(source_filename), f"Source file does not exist: {source_filename}"
        assert os.path.exists(target_filename), f"Target file does not exist: {target_filename}"
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)
        # ipdb.set_trace()
        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


# Testing the dataset
if __name__ == '__main__':
    dataset = MyDataset()
    print(len(dataset))

    for i in range(3):
        item = dataset[i]
        print(item['jpg'].shape, item['txt'], item['hint'].shape)
        print()