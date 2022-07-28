import torch
import torch.nn as nn

class CalibModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, stride=1),
            nn.ELU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3, stride=1),
            nn.ELU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=30, out_channels=60, kernel_size=3, stride=1),
            nn.ELU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=60, out_channels=60, kernel_size=3, stride=1),
            nn.ELU(),
            nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(10560, 2640),
            nn.ELU(),
            nn.Linear(2640, 512),
            nn.ELU(),
            nn.Linear(512, 2),
        )

    def forward(self,x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1) #flatten
        x = self.classifier(x)
        return x