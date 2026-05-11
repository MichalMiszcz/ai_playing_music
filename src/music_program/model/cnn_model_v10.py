"""
Model bazującego na sieci konwolucyjnej. Dane uczące to obrazy zawierające tylko jedną pięciolinię oraz skrócone MIDI.
"""

import torch
import torch.nn as nn
from torchvision import models

from src.music_program.utils.global_variables import *

kernel_y = int(HEIGHT / 3)
kernel_x = int(WIDTH / 32)
stride = kernel_x // 3


class MusicModel(nn.Module):
    def __init__(self, features_number, linear_dim, max_series_len):
        super(MusicModel, self).__init__()
        self.max_series_len = max_series_len
        self.features_number = features_number

        out_channels = features_number
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_y, kernel_x),
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # out_channels = out_channels*2
        self.conv2 = nn.Conv2d(in_channels=int(out_channels), out_channels=out_channels, kernel_size=3, stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #
        # # out_channels *= 2
        # self.conv3 = nn.Conv2d(in_channels=int(out_channels), out_channels=out_channels, kernel_size=3, stride=1,
        #                        padding=1)
        # self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((1, self.max_series_len))
        self.fc = nn.Linear(out_channels * self.max_series_len, 2048)
        self.dropout = nn.Dropout(p=0.3)
        self.linear_1 = nn.Linear(2048, linear_dim)
        self.linear_2 = nn.Linear(linear_dim, self.max_series_len)

        # self.linear_1 = nn.Linear(512, linear_dim*2)
        # self.linear_2 = nn.Linear(linear_dim*2, linear_dim)
        # self.linear_3 = nn.Linear(linear_dim, self.max_series_len)
        self.hard_tanh = nn.Hardtanh(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.max_pool(x)
        #
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)

        x = self.adaptive_max_pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear_2(x)
        # x = self.dropout(x)
        # x = self.relu(x)
        #
        # x = self.linear_3(x)

        x = self.hard_tanh(x)

        return x
