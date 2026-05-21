"""
Model bazującego na sieci konwolucyjnej. Dane uczące to obrazy zawierające tylko jedną pięciolinię oraz skrócone MIDI.
"""

import torch
import torch.nn as nn
from torchvision import models


class MusicModel(nn.Module):
    def __init__(self, features_number, hidden_dim, max_series_len):
        super(MusicModel, self).__init__()
        self.max_series_len = max_series_len
        self.features_number = features_number

        out_channels = features_number
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=7,
                               stride=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        out_channels = out_channels * 2
        self.conv2 = nn.Conv2d(in_channels=int(out_channels / 2), out_channels=out_channels, kernel_size=3, stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        out_channels = out_channels * 2
        self.conv3 = nn.Conv2d(in_channels=int(out_channels / 2), out_channels=out_channels, kernel_size=3, stride=1,
                               padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        out_channels = out_channels * 2
        self.conv4 = nn.Conv2d(in_channels=int(out_channels / 2), out_channels=out_channels, kernel_size=3, stride=1,
                               padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc_shrink = nn.Linear(7 * 21, self.max_series_len)

        self.rnn = nn.GRU(input_size=256, hidden_size=hidden_dim,
                          num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)

        self.fc = nn.Linear(hidden_dim * 2, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)


        x = torch.flatten(x, 2)
        x = self.fc_shrink(x)
        x = x.permute(0, 2, 1)

        x, _ = self.rnn(x)

        x = self.fc(x)

        return x