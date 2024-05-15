import json
import cv2
import numpy as np
import ipdb
import os
from torch.utils.data import Dataset
from annotator.util import resize_image, HWC3
from torchvision import transforms
from PIL import Image


# TODO: add more data augmentation for source images.
class Quilt(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/quilt_1M_prompt.json', 'rt') as f:
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

        # source_filename = os.path.join(self.data_dir, item['source'])
        source_filename = os.path.join(self.data_dir, os.path.splitext(item['source'])[0] + '.npy')
        target_filename = os.path.join(self.data_dir, item['target'])
        # ipdb.set_trace()
        assert os.path.exists(source_filename), f"Source file does not exist: {source_filename}"
        assert os.path.exists(target_filename), f"Target file does not exist: {target_filename}"
        prompt = item['prompt']
        source = HWC3(np.load(source_filename))
        
        target = cv2.imread(target_filename)
        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class ChaoyangDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/chaoyang/prompt.json', 'rt') as f:
            self.data = json.load(f)
        self.data_dir = './training/chaoyang'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = os.path.join(self.data_dir, os.path.splitext(item['source'])[0] + '.npy')
        target_filename = os.path.join(self.data_dir, item['target'])
        # ipdb.set_trace()
        assert os.path.exists(source_filename), f"Source file does not exist: {source_filename}"
        assert os.path.exists(target_filename), f"Target file does not exist: {target_filename}"
        prompt = item['prompt']
        source = HWC3(np.load(source_filename))
        
        target = cv2.imread(target_filename)
        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class ChaoyangTestDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/chaoyang/test_prompt.json', 'rt') as f:
            self.data = json.load(f)
        self.data_dir = './training/chaoyang'
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        name = os.path.basename(item['source']) # name: 537688_1-IMG005x003-3_70_190.npy
        source_filename = os.path.join(self.data_dir, item['source']) 
        assert os.path.exists(source_filename), f"Source file does not exist: {source_filename}"
        prompt = item['prompt']
        source = HWC3(np.load(source_filename))
        
        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # source = source.astype(np.float32) / 255.0

        source = self.to_tensor(Image.fromarray(source))
        if len(source.shape) == 2:
            source = source.convert('RGB')

        return dict(txt=prompt, hint=source, name=name)

# Testing the dataset
if __name__ == '__main__':
    dataset = ChaoyangDataset() 
    print(len(dataset))

    for i in range(3):
        item = dataset[i]
        print(item['jpg'].shape, item['txt'], item['hint'].shape)
        print()