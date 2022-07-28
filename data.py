import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((218,291)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=.5),
        transforms.RandomInvert(p=.5),
        transforms.Normalize(0.2311, 0.1354),
    ]),

    'test': transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((218,291)),
        transforms.ToTensor(),
        transforms.Normalize(0.2311, 0.1354),

    ])
}


class CalibDataset(Dataset):
    def __init__(self,train=True):
        self.train = train
        self.df = pd.read_csv('data/' + ('train' if train else 'test') + '.csv')
        self.transforms = data_transforms['train' if train else 'test']


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row['frame_path']
        pitch = row['std_pitch']
        yaw = row['std_yaw']

        img = Image.open(path)
        x = data_transforms['train' if self.train else 'test'](img).float()
        y = torch.Tensor([pitch, yaw])

        return x,y



