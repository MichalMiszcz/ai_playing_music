import torch
import torch.nn as nn
import torchvision.models as models


class CNNRNNClassificationModel(nn.Module):
    def __init__(self, hidden_dim=256, rnn_layers=2, max_seq_len=100, max_series_len=50):
        super(CNNRNNClassificationModel, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_series_len = max_series_len

        # Słowniki (ilość unikalnych klas)
        self.num_notes = 8  # Nuty 0-7
        self.num_times = 6  # 6 unikalnych wartości delta time

        # Zamiast podawać surowe liczby (np. 5040), używamy Embeddings (jak słowa w NLP)
        self.note_emb = nn.Embedding(self.num_notes, 32)
        self.time_emb = nn.Embedding(self.num_times, 32)

        # CNN (zamrożony lub nie, zostawiam Twój kod)
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        cnn_resnet_weight = self.cnn.conv1.weight
        self.cnn.conv1 = nn.Conv2d(3, 64, kernel_size=76, stride=16, padding=0, bias=False)
        self.cnn.conv1.weight = cnn_resnet_weight
        self.cnn.fc = nn.Linear(512, hidden_dim)

        # RNN: wejście to teraz wektor nuty (32) + wektor czasu (32) + context CNN (hidden_dim)
        self.rnn = nn.LSTM(input_size=32 + 32 + hidden_dim,
                           hidden_size=hidden_dim,
                           num_layers=rnn_layers,
                           dropout=0.3,  # Zwiększony dropout dla regularyzacji
                           batch_first=True)

        # Dwie osobne głowice klasyfikujące (Output)
        self.fc_note = nn.Linear(hidden_dim, self.num_notes)
        self.fc_time = nn.Linear(hidden_dim, self.num_times)

    def forward(self, x, target=None, teacher_ratio=None, temperature=1.0):
        batch_size = x.size(0)
        features = self.cnn(x).view(batch_size, -1)  # Kształt: (batch, hidden_dim)

        use_teacher_learning = torch.rand(1).item()

        # --- TRYB TRENINGU (Teacher Forcing) ---
        if target is not None and teacher_ratio is not None and use_teacher_learning <= teacher_ratio:
            # Zakładamy, że target ma kształt (batch, seq_len, 2) i zawiera INDEKSY (typu long/int)
            # target[:, :, 0] to nuty (0-7), target[:, :, 1] to czasy (0-5)

            note_in = self.note_emb(target[:, :-1, 0])
            time_in = self.time_emb(target[:, :-1, 1])

            # Puste wektory na start sekwencji
            start_note = torch.zeros(batch_size, 1, 32).to(x.device)
            start_time = torch.zeros(batch_size, 1, 32).to(x.device)

            input_note = torch.cat([start_note, note_in], dim=1)
            input_time = torch.cat([start_time, time_in], dim=1)

            seq_len = input_note.size(1)
            context = features.unsqueeze(1).expand(-1, seq_len, -1)

            rnn_input = torch.cat([input_note, input_time, context], dim=-1)

            output, _ = self.rnn(rnn_input)

            # Zwracamy "surowe" logity (brak Hardsigmoid!)
            note_logits = self.fc_note(output)
            time_logits = self.fc_time(output)

            return note_logits, time_logits

        # --- TRYB GENEROWANIA (Inference / Brak Nauczyciela) ---
        else:
            note_logits_seq = []
            time_logits_seq = []

            # Inicjalizacja pustymi osadzeniami
            curr_note_emb = torch.zeros(batch_size, 1, 32).to(x.device)
            curr_time_emb = torch.zeros(batch_size, 1, 32).to(x.device)
            hidden = None
            context_step = features.unsqueeze(1)

            for _ in range(self.max_series_len):
                rnn_input = torch.cat([curr_note_emb, curr_time_emb, context_step], dim=-1)

                output, hidden = self.rnn(rnn_input, hidden)

                # Surowe prawdopodobieństwa
                note_logits = self.fc_note(output)
                time_logits = self.fc_time(output)

                note_logits_seq.append(note_logits)
                time_logits_seq.append(time_logits)

                # Samplowanie z temperaturą (wprowadza "kreatywność")
                note_probs = torch.softmax(note_logits.squeeze(1) / temperature, dim=-1)
                time_probs = torch.softmax(time_logits.squeeze(1) / temperature, dim=-1)

                # Zamiast wybierać zawsze najwyższe, losujemy wg prawdopodobieństwa
                next_note = torch.multinomial(note_probs, 1)  # Kształt (batch, 1)
                next_time = torch.multinomial(time_probs, 1)  # Kształt (batch, 1)

                # Konwersja wygenerowanych indeksów z powrotem na wektory dla kolejnego kroku
                curr_note_emb = self.note_emb(next_note)
                curr_time_emb = self.time_emb(next_time)

            note_logits_out = torch.cat(note_logits_seq, dim=1)
            time_logits_out = torch.cat(time_logits_seq, dim=1)

            # Możesz zwrócić logity do obliczenia błędu lub wygenerowane sekwencje do posłuchania
            return note_logits_out, time_logits_out