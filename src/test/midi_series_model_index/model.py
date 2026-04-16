"""
Implementacja modelu sekwencja-sekwencja (MIDI — ciąg nut).

Wyjściowa sekwencja wygląda w następujący sposób:
[64, 64, 65, 64, 64, 64, 64, 62, 62, 62, 62, 62, 62, ... , 60]
"""

import random

import torch
import torch.nn as nn


class ModelLSTM(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=64, input_dim=3, rnn_layers=2, max_seq_len=90, max_series_len=450, vocab_size=40) -> None:
        super(ModelLSTM, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_series_len = max_series_len
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.decoder = nn.LSTM(input_size=embedding_dim + hidden_dim, hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.encoder_linear = nn.Linear(max_seq_len * hidden_dim, hidden_dim)
        self.proj_h = nn.Linear(hidden_dim, hidden_dim * rnn_layers)
        self.proj_c = nn.Linear(hidden_dim, hidden_dim * rnn_layers)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, target=None, teacher_ratio=None):
        batch_size = x.size(0)

        encoder_output, _ = self.encoder(x)

        encoder_output = encoder_output.reshape(batch_size, -1)
        features = self.encoder_linear(encoder_output)

        # nowe generowanie sekwencji
        use_teacher_learning = torch.rand(1).item()
        if target is not None and teacher_ratio is not None and use_teacher_learning <= teacher_ratio:
            target_embedded = self.embedding(target)

            sos_token = torch.zeros(batch_size, 1, self.embedding.embedding_dim).to(x.device)
            input_seq = torch.cat([sos_token, target_embedded[:, :-1, :]], dim=1)

            # dodane
            seq_len = input_seq.size(1)
            context = features.unsqueeze(1).expand(-1, seq_len, -1)
            decoder_input = torch.cat([input_seq, context], dim=-1)
            # do tąd

            output, _ = self.decoder(decoder_input)
            # output, _ = self.rnn(input_seq, (h0, c0))
            output = self.dropout(output)
            output = self.linear(output)

            # output = output.view(output.size(0), output.size(1))
            return output
        else:
            output_seq = []
            input_note = torch.zeros(batch_size, 1, self.embedding.embedding_dim).to(x.device)
            hidden = None  # (h0, c0)

            # dodane
            context_step = features.unsqueeze(1)

            # var_sigmoid = torch.sigmoid
            var_tanh = torch.tanh
            torch_cat = torch.cat
            for _ in range(self.max_series_len):
                # dodane
                decoder_input = torch_cat([input_note, context_step], dim=-1)

                # output, hidden = self.rnn(input_note, hidden)
                output, hidden = self.decoder(decoder_input, hidden)
                output = self.dropout(output)
                output = self.linear(output)
                output_seq.append(output)

                predicted_idx = output.argmax(dim=-1)
                input_note = self.embedding(predicted_idx)

            output = torch.cat(output_seq, dim=1)
            # output = output.view(output.size(0), output.size(1))

            return output
