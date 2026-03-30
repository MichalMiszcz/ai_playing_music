"""
Implementacja modelu obraz-sekwencja o jednowymiarowym wyjściu.

Wyjściowa sekwencja wygląda w następujący sposób:
[64, 64, 65, 64, 64, 64, 64, 62, 62, 62, 62, 62, 62, ... , 60]
"""

import random

import torch
import torch.nn as nn

from src.music_program.utils.resnet_encoder import Encoder, BasicBlockEnc


class CNNRNNModel(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=1024, output_dim=1, rnn_layers=3, max_seq_len=100):
        super(CNNRNNModel, self).__init__()
        self.max_seq_len = max_seq_len
        # self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # self.cnn = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.cnn = Encoder(BasicBlockEnc, [1, 0, 0, 0])
        # self.cnn.load_state_dict(torch.load('src/wagi_enkodera.pth'))
        # for param in self.cnn.parameters():
        #     param.requires_grad = False

        self.fc = nn.Linear(64, hidden_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.rnn = nn.LSTM(input_size=output_dim+hidden_dim, hidden_size=hidden_dim, num_layers=rnn_layers, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.proj_h = nn.Linear(hidden_dim, hidden_dim * rnn_layers)
        self.proj_c = nn.Linear(hidden_dim, hidden_dim * rnn_layers)

        self.output_dim = output_dim

    def forward(self, x, target=None, teacher_ratio=None):
        batch_size = x.size(0)

        # odwzorowanie ResNet
        features = self.cnn(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        features = self.fc(features)
        features = features.view(batch_size, -1)

        # h0 = self.proj_h(features).view(batch_size, self.rnn.num_layers, -1).transpose(0, 1).contiguous()
        # c0 = self.proj_c(features).view(batch_size, self.rnn.num_layers, -1).transpose(0, 1).contiguous()

        # nowe generowanie sekwencji
        use_teacher_learning = random.random()
        if target is not None and teacher_ratio is not None and use_teacher_learning <= teacher_ratio:
            target = target.view(target.size(0), target.size(1), 1)
            input_seq = torch.cat([torch.zeros(batch_size, 1, self.output_dim).to(x.device), target[:, :-1, :]], dim=1)

            # dodane
            seq_len = input_seq.size(1)
            context = features.unsqueeze(1).expand(-1, seq_len, -1)
            rnn_input = torch.cat([input_seq, context], dim=-1)
            # do tąd

            output, _ = self.rnn(rnn_input)
            # output, _ = self.rnn(input_seq, (h0, c0))
            output = self.linear(output)

            # output = torch.sigmoid(output)
            output = torch.tanh(output)

            output = output.view(output.size(0), output.size(1))
            return output
        else:
            output_seq = []
            input_note = torch.zeros(batch_size, 1, self.output_dim).to(x.device)
            hidden = None  # (h0, c0)

            # dodane
            context_step = features.unsqueeze(1)

            # var_sigmoid = torch.sigmoid
            var_tanh = torch.tanh
            for _ in range(self.max_seq_len):
                # dodane
                rnn_input = torch.cat([input_note, context_step], dim=-1)

                # output, hidden = self.rnn(input_note, hidden)
                output, hidden = self.rnn(rnn_input, hidden)
                output = self.linear(output)
                output = var_tanh(output)

                output_seq.append(output)
                input_note = output

            output = torch.cat(output_seq, dim=1)
            output = output.view(output.size(0), output.size(1))

            return output