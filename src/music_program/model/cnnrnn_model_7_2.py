"""
Implementacja modelu obraz-sekwencja o jednowymiarowym wyjściu.

Wyjściowa sekwencja wygląda w następujący sposób:
[1, 12, 42, 42, 2, ...]
"""

import torch
import torch.nn as nn
from torchvision import models

from src.music_program.utils.global_variables import *


class CNNRNNModel(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=1024, emb_dim_note=32, emb_dim_delta_time=32, output_dim=1, rnn_layers=3, max_seq_len=100, max_series_len=50):
        super(CNNRNNModel, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_series_len = max_series_len

        self.num_notes = NUM_NOTES
        self.num_times = NUM_DELTA_TIME

        self.emb_dim_note = emb_dim_note
        self.emb_dim_delta_time = emb_dim_delta_time
        self.note_emb = nn.Embedding(self.num_notes, emb_dim_note)
        self.delta_time_emb = nn.Embedding(self.num_times, emb_dim_delta_time)

        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.cnn.conv1 = nn.Conv2d(input_channels, 64, kernel_size=76, stride=16, padding=0, bias=False)

        # cnn_resnet_weight = self.cnn.conv1.weight
        # self.cnn.conv1 = nn.Conv2d(3, 64, kernel_size=76, stride=16, padding=0, bias=False)
        # self.cnn.conv1.weight = cnn_resnet_weight

        self.cnn.fc = nn.Linear(512, hidden_dim)

        self.rnn = nn.LSTM(input_size=emb_dim_note + emb_dim_delta_time + hidden_dim, hidden_size=hidden_dim, num_layers=rnn_layers, dropout=0.3, batch_first=True)
        self.fc_note = nn.Linear(hidden_dim, self.num_notes)
        self.fc_time = nn.Linear(hidden_dim, self.num_times)

    def forward(self, x, target=None, teacher_ratio=None, temperature_note=1.0, temperature_time=1.0):
        batch_size = x.size(0)

        features = self.cnn(x).view(batch_size, -1)

        use_teacher_learning = torch.rand(1).item()
        if target is not None and teacher_ratio is not None and use_teacher_learning <= teacher_ratio:
            note_in = self.note_emb(target[:, :-1, 0])
            time_in = self.delta_time_emb(target[:, :-1, 1])

            start_note = torch.zeros(batch_size, 1, self.emb_dim_note).to(x.device)
            start_time = torch.zeros(batch_size, 1, self.emb_dim_delta_time).to(x.device)

            input_note = torch.cat([start_note, note_in], dim=1)
            input_time = torch.cat([start_time, time_in], dim=1)

            seq_len = input_note.size(1)
            context = features.unsqueeze(1).expand(-1, seq_len, -1)
            rnn_input = torch.cat([input_note, input_time, context], dim=-1)

            output, _ = self.rnn(rnn_input)
            note_logits = self.fc_note(output)
            time_logits = self.fc_time(output)

            return note_logits, time_logits
        else:
            note_logits_seq = []
            time_logits_seq = []

            curr_note_emb = torch.zeros(batch_size, 1, self.emb_dim_note).to(x.device)
            curr_time_emb = torch.zeros(batch_size, 1, self.emb_dim_delta_time).to(x.device)
            hidden = None
            context_step = features.unsqueeze(1)

            torch_cat = torch.cat
            for _ in range(self.max_series_len):
                rnn_input = torch_cat([curr_note_emb, curr_time_emb, context_step], dim=-1)

                output, hidden = self.rnn(rnn_input, hidden)
                note_logits = self.fc_note(output)
                time_logits = self.fc_time(output)

                note_logits_seq.append(note_logits)
                time_logits_seq.append(time_logits)

                note_probs = torch.softmax(note_logits.squeeze(1) / temperature_note, dim=-1)
                time_probs = torch.softmax(time_logits.squeeze(1) / temperature_time, dim=-1)

                next_note = torch.multinomial(note_probs, 1)
                next_time = torch.multinomial(time_probs, 1)

                curr_note_emb = self.note_emb(next_note)
                curr_time_emb = self.delta_time_emb(next_time)

            note_logits_out = torch.cat(note_logits_seq, dim=1)
            time_logits_out = torch.cat(time_logits_seq, dim=1)

            return note_logits_out, time_logits_out