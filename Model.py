import torch
import torch.nn as nn
import torchsummary
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = self.make_block(in_channels=3, out_channels=16)
        self.conv2 = self.make_block(in_channels=16, out_channels=32)
        self.conv3 = self.make_block(in_channels=32, out_channels=64)
        self.conv4 = self.make_block(in_channels=64, out_channels=64)
        # self.conv5 = self.make_block(in_channels=64, out_channels=64)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=9216, out_features=4608),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=4608, out_features=2304),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=2304, out_features=2)
        )
    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
