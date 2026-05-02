"""
Implementacja modelu obraz-sekwencja o jednowymiarowym wyjściu.

Wyjściowa sekwencja wygląda w następujący sposób:
[(64, 20160), (72, 20160), (72, 5040), (65, 10080), (64, 5040), ...]
"""

import torch
import torch.nn as nn
from torchvision import models


class CNNRNNModel(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=1024, output_dim=1, rnn_layers=3, max_seq_len=100, max_series_len=50):
        super(CNNRNNModel, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_series_len = max_series_len
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # self.cnn = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        # cnn_resnet_weight = self.cnn.conv1.weight
        # self.cnn.conv1 = nn.Conv2d(input_channels, 64, kernel_size=1, stride=1, padding=0, bias=False)
        # self.cnn.conv1.weight.data = cnn_resnet_weight.mean(dim=1, keepdim=True)

        stride = 16
        kernel_size = 76 # nuta ma rozmiar 8 na 8 pikseli, a mamy 8 nut, które nachodzą na siebie, dodatkowo dodamy margines dla stride, oraz dodatkowy
        padding = 0

        print(f"Kernel size: {kernel_size}, stride: {stride}, padding: {padding}")

        cnn_resnet_weight = self.cnn.conv1.weight
        self.cnn.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.cnn.conv1.weight = cnn_resnet_weight

        # self.cnn.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.cnn.fc = nn.Linear(512, hidden_dim)
        # self.cnn.fc = nn.Linear(2048, hidden_dim)

        self.rnn = nn.LSTM(input_size=output_dim+hidden_dim, hidden_size=hidden_dim, num_layers=rnn_layers, dropout=0.3, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.proj_h = nn.Linear(hidden_dim, hidden_dim * rnn_layers)
        self.proj_c = nn.Linear(hidden_dim, hidden_dim * rnn_layers)

        self.output_dim = output_dim

    def forward(self, x, target=None, teacher_ratio=None):
        batch_size = x.size(0)

        features = self.cnn(x).view(batch_size, -1)

        # h0 = self.proj_h(features).view(batch_size, self.rnn.num_layers, -1).transpose(0, 1).contiguous()
        # c0 = self.proj_c(features).view(batch_size, self.rnn.num_layers, -1).transpose(0, 1).contiguous()

        use_teacher_learning = torch.rand(1).item()
        if target is not None and teacher_ratio is not None and use_teacher_learning <= teacher_ratio:
            input_seq = torch.cat([torch.zeros(batch_size, 1, self.output_dim).to(x.device), target[:, :-1, :]], dim=1)

            seq_len = input_seq.size(1)
            context = features.unsqueeze(1).expand(-1, seq_len, -1)
            rnn_input = torch.cat([input_seq, context], dim=-1)

            output, _ = self.rnn(rnn_input)
            # output, _ = self.rnn(input_seq, (h0, c0))
            output = self.linear(output)

            # output = torch.sigmoid(output)
            output = torch.nn.Hardsigmoid()(output)
            # output = torch.tanh(output)
            return output
        else:
            output_seq = []
            input_note = torch.zeros(batch_size, 1, self.output_dim).to(x.device)
            hidden = None  # (h0, c0)

            # dodane
            context_step = features.unsqueeze(1)

            # var_sigmoid = torch.sigmoid
            var_sigmoid = torch.nn.Hardsigmoid()
            # var_tanh = torch.tanh
            torch_cat = torch.cat
            for _ in range(self.max_series_len):
                rnn_input = torch_cat([input_note, context_step], dim=-1)

                # output, hidden = self.rnn(input_note, hidden)
                output, hidden = self.rnn(rnn_input, hidden)
                output = self.linear(output)
                output = var_sigmoid(output)

                output_seq.append(output)
                input_note = output

            output = torch.cat(output_seq, dim=1)

            return output