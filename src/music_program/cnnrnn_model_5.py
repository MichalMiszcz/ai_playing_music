import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock


class ResNet4(ResNet):

    def __init__(self, block = BasicBlock, layers = [1, 1, 1, 1]):
        super().__init__(block, layers)
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class CNNRNNModel(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=1024, output_dim=3, rnn_layers=3, max_seq_len=100):
        super(CNNRNNModel, self).__init__()
        self.max_seq_len = max_seq_len
        self.cnn = ResNet4()
        self.cnn.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.cnn.maxpool = nn.Identity()
        self.cnn.fc = nn.Linear(64, hidden_dim)
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