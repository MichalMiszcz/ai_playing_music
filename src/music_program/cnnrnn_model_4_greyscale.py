import random

import torch
import torch.nn as nn
from torchvision import models

class CNNRNNModel(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=1024, output_dim=3, rnn_layers=3, max_seq_len=100):
        super(CNNRNNModel, self).__init__()
        self.max_seq_len = max_seq_len
        # self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # self.cnn = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # self.cnn.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(32, 12), stride=(2, 1), padding=0, bias=False)
        # self.cnn.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(32, 12), stride=(4, 2), padding=8, bias=False)
        self.cnn.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(36, 12), stride=(4, 2), padding=8, bias=False)
        # self.cnn.fc = nn.Linear(512, hidden_dim) # ResNet34 and 18
        self.cnn.fc = nn.Linear(2048, hidden_dim) # ResNet50
        # self.cnn.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.rnn = nn.LSTM(input_size=output_dim, hidden_size=hidden_dim, num_layers=rnn_layers, dropout=0.33, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.proj_h = nn.Linear(hidden_dim, hidden_dim * rnn_layers)
        self.proj_c = nn.Linear(hidden_dim, hidden_dim * rnn_layers)

        self.output_dim = output_dim

    def forward(self, x, target=None):
        batch_size = x.size(0)
        features = self.cnn(x).view(batch_size, -1)
        h0 = self.proj_h(features).view(batch_size, self.rnn.num_layers, -1).transpose(0, 1).contiguous()
        c0 = self.proj_c(features).view(batch_size, self.rnn.num_layers, -1).transpose(0, 1).contiguous()

        if target is not None:
            input_seq = torch.cat([torch.zeros(batch_size, 1, self.output_dim).to(x.device), target[:, :-1, :]], dim=1)
            output, _ = self.rnn(input_seq, (h0, c0))
            output = self.linear(output)
            output = torch.sigmoid(output)
            return output
        else:
            output_seq = []
            input_note = torch.zeros(batch_size, 1, self.output_dim).to(x.device)
            hidden = (h0, c0)
            for _ in range(self.max_seq_len):
                output, hidden = self.rnn(input_note, hidden)
                output = self.linear(output)
                output = torch.sigmoid(output)
                output_seq.append(output)
                input_note = output
            return torch.cat(output_seq, dim=1)