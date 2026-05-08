"""
Model bazującego na sieci konwolucyjnej. Dane uczące to obrazy zawierające tylko jedną pięciolinię oraz skrócone MIDI.
"""

import torch
import torch.nn as nn
from torchvision import models

from src.music_program.utils.global_variables import *

kernel_y = int(HEIGHT/3)
kernel_x = int(WIDTH/32)
stride = kernel_x // 3

class MusicModel(nn.Module):
    def __init__(self, features_number, max_series_len):
        super(MusicModel, self).__init__()
        self.max_series_len = max_series_len
        self.features_number = features_number

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=features_number, kernel_size=(kernel_y, kernel_x), stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(features_number)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=features_number, out_channels=features_number, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(features_number)

        # self.avg_pool = nn.AdaptiveAvgPool2d((1, -1))
        # self.fc = nn.Linear(features_number * self.max_series_len * 4, 64)
        self.fc = nn.Linear(features_number * int(((HEIGHT + 2 * 1 - kernel_x)/stride + 1)) , 64)
        self.linear = nn.Linear(64, self.max_series_len)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        x = self.linear(x)
        x = torch.sigmoid(x)

        return x
