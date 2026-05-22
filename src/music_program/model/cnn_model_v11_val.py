"""
Model bazującego na sieci konwolucyjnej. Dane uczące to obrazy zawierające tylko jedną pięciolinię oraz dwuwymiarowe MIDI.
"""

import torch
import torch.nn as nn
from torchvision import models


class MusicModel(nn.Module):
    def __init__(self, features_number, hidden_dim, max_series_len):
        super(MusicModel, self).__init__()
        self.max_series_len = max_series_len
        self.features_number = features_number

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, None))

        self.fc_shrink = nn.Linear(85, self.max_series_len)

        self.rnn = nn.GRU(input_size=256, hidden_size=hidden_dim,
                          num_layers=2, bidirectional=True, batch_first=True, dropout=0.4)

        self.note_fc = nn.Linear(hidden_dim * 2, 9)
        self.time_fc = nn.Linear(hidden_dim * 2, 6)

    def forward(self, x):
        x = self.cnn(x)
        x = self.adaptive_pool(x)

        x = x.squeeze(2)
        x = self.fc_shrink(x)
        x = x.permute(0, 2, 1)

        x, _ = self.rnn(x)

        note = self.note_fc(x)
        time = self.time_fc(x)

        return note, time