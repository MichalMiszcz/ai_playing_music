import torch
import torch.nn as nn
from torchvision import models

class CNNRNNModel(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=1024, output_dim=4, rnn_layers=4, max_seq_len=100):
        super(CNNRNNModel, self).__init__()
        self.max_seq_len = max_seq_len
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.conv1 = nn.Conv2d(input_channels, 64, kernel_size=25, stride=1, padding=5, bias=True) #może bias na true
        self.cnn.fc = nn.Linear(512, hidden_dim)
        self.cnn.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.rnn = nn.LSTM(input_size=output_dim, hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.proj_h = nn.Linear(hidden_dim, hidden_dim * rnn_layers)
        self.proj_c = nn.Linear(hidden_dim, hidden_dim * rnn_layers)

    def forward(self, x, target=None):
        batch_size = x.size(0)
        features = self.cnn(x).view(batch_size, -1)
        h0 = self.proj_h(features).view(batch_size, self.rnn.num_layers, -1).transpose(0, 1).contiguous()
        c0 = self.proj_c(features).view(batch_size, self.rnn.num_layers, -1).transpose(0, 1).contiguous()

        if target is not None:
            input_seq = torch.cat([torch.zeros(batch_size, 1, 4).to(x.device), target[:, :-1, :]], dim=1)
            output, _ = self.rnn(input_seq, (h0, c0))
            output = self.linear(output)
            output = torch.sigmoid(output)
            return output
        else:
            output_seq = []
            input_note = torch.zeros(batch_size, 1, 4).to(x.device)
            hidden = (h0, c0)
            for _ in range(self.max_seq_len):
                output, hidden = self.rnn(input_note, hidden)
                output = self.linear(output)
                output = torch.sigmoid(output)
                output_seq.append(output)
                input_note = output
            return torch.cat(output_seq, dim=1)