import pandas as pd
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = {'train': tv.transforms.Compose([

            tv.transforms.ToPILImage(),
            # tv.transforms.Resize((227,227)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)]),
            'val': tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                # tv.transforms.Resize((227,227)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)])}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data.iloc[index, 0]
        label = np.array(self.data.iloc[index, 1:]).astype('float')

        image = imread(image_path)
        image = gray2rgb(image)
        image = self._transform[self.mode](image)
        return torch.Tensor(image), torch.Tensor(label)



