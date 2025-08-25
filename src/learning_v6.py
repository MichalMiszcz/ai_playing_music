import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import mido

image_root = "data/images"
midi_root = "data/processed_midi"
selected_image_path = "data/images/albeniz/alb_esp1/alb_esp1-1.png"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Image transform for grayscale
image_transform = transforms.Compose([
    transforms.Resize((992, 1402)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class MusicImageDataset(Dataset):
    def __init__(self, image_root, midi_root, image_transform=None, max_seq_len=100, max_midi_files=100):
        self.image_root = image_root
        self.midi_root = midi_root
        self.image_transform = image_transform if image_transform else transforms.ToTensor()
        self.max_seq_len = max_seq_len
        self.image_files = []
        self.midi_features = {}

        for author in os.listdir(image_root)[:max_midi_files]:
            img_path = os.path.join(image_root, author, "image.png")
            midi_path = os.path.join(midi_root, author, "music.mid")
            if os.path.exists(img_path) and os.path.exists(midi_path):
                self.image_files.append((img_path, author))
                midi_seq = extract_notes_from_midi(midi_path)
                if len(midi_seq) > max_seq_len:
                    midi_seq = midi_seq[:max_seq_len]
                else:
                    midi_seq += [(0, 0, 0)] * (max_seq_len - len(midi_seq))
                self.midi_features[author] = midi_seq

        all_time_deltas = [td for seq in self.midi_features.values() for td, _, _ in seq if td > 0]
        all_durations = [d for seq in self.midi_features.values() for _, _, d in seq if d > 0]
        self.max_time_delta = max(all_time_deltas) if all_time_deltas else 1
        self.max_duration = max(all_durations) if all_durations else 1

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path, author = self.image_files[idx]
        image = Image.open(img_path).convert('L')
        if self.image_transform:
            image = self.image_transform(image)

        midi_seq = self.midi_features[author]
        normalized_seq = [
            (time_delta / self.max_time_delta, note / 127.0, duration / self.max_duration)
            for time_delta, note, duration in midi_seq
        ]
        midi_tensor = torch.tensor(normalized_seq, dtype=torch.float32)
        return image, midi_tensor


def extract_notes_from_midi(midi_path):
    mid = mido.MidiFile(midi_path)
    right_hand_notes = []
    current_time = 0
    note_starts = {}

    for msg in mid:
        if not msg.is_meta:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_starts[msg.note] = current_time
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in note_starts:
                    start_time = note_starts.pop(msg.note)
                    duration = current_time - start_time
                    right_hand_notes.append((start_time, msg.note, duration))

    right_hand_notes.sort(key=lambda x: x[0])
    sequence = []
    prev_time = 0
    for start_time, note, duration in right_hand_notes:
        time_delta = start_time - prev_time
        sequence.append((time_delta, note, duration))
        prev_time = start_time
    return sequence


class CNNRNNModel(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=1024, output_dim=3, rnn_layers=2, max_seq_len=100):
        super(CNNRNNModel, self).__init__()
        self.max_seq_len = max_seq_len
        self.rnn = nn.LSTM(input_size=output_dim, hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True)

        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Identity()
        self.cnn.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj_h = nn.Linear(512, hidden_dim * rnn_layers)
        self.proj_c = nn.Linear(512, hidden_dim * rnn_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, target=None):
        batch_size = x.size(0)
        features = self.cnn(x).view(batch_size, -1)
        h0 = self.proj_h(features).view(batch_size, self.rnn.num_layers, 1024).transpose(0, 1).contiguous()
        c0 = self.proj_c(features).view(batch_size, self.rnn.num_layers, 1024).transpose(0, 1).contiguous()

        if target is not None:
            input_seq = torch.cat([torch.zeros(batch_size, 1, 3).to(x.device), target[:, :-1, :]], dim=1)
            output, _ = self.rnn(input_seq, (h0, c0))
            output = self.linear(output)
            output = torch.sigmoid(output)
            return output
        else:
            output_seq = []
            input_note = torch.zeros(batch_size, 1, 3).to(x.device)
            hidden = (h0, c0)
            for _ in range(self.max_seq_len):
                output, hidden = self.rnn(input_note, hidden)
                output = self.linear(output)
                output = torch.sigmoid(output)
                output_seq.append(output)
                input_note = output
            return torch.cat(output_seq, dim=1)


def generate_midi_from_selected_image(model, dataset, image_path, output_midi_path, device='cuda'):
    model.eval()
    image = Image.open(image_path).convert('L')
    image = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        predicted_sequence = output[0].cpu().numpy().tolist()
        predicted_sequence = [
            (int(t * dataset.max_time_delta + 0.5), int(p * 127 + 0.5), int(d * dataset.max_duration + 0.5))
            for t, p, d in predicted_sequence
        ]
        sequence_to_midi(predicted_sequence, output_midi_path)


def sequence_to_midi(sequence, output_midi_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    events = []
    current_time = 0
    for time_delta, note, duration in sequence:
        start_time = current_time + time_delta
        end_time = start_time + duration
        events.append((start_time, 'note_on', note))
        events.append((end_time, 'note_off', note))
        current_time = start_time

    events.sort(key=lambda x: x[0])

    prev_time = 0
    for time, event_type, note in events:
        delta = int(time - prev_time)
        if event_type == 'note_on':
            track.append(mido.Message('note_on', note=note, velocity=64, time=delta))
        else:
            track.append(mido.Message('note_off', note=note, velocity=0, time=delta))
        prev_time = time

    mid.save(output_midi_path)


def train_model(model, dataloader, epochs=50, device=device, learning_rate=0.0005):
    learning_data = []

    model = model.to(device)
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (images, midi_batch) in enumerate(dataloader):
            images = images.to(device)
            midi_batch = midi_batch.to(device)

            optimizer.zero_grad()
            output = model(images, midi_batch)
            loss = criterion_mse(output, midi_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.6f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        learning_data.append((epoch, avg_loss))

    torch.save(model.state_dict(), 'model_mini.pth')
    print("Model saved as 'model_mini.pth'")
    return learning_data


def generate_chart(data):
    epochs = [t[0] for t in data]
    avg_losses = [t[1] for t in data]

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, avg_losses, marker='o', linestyle='-', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    max_seq_len = 25

    dataset = MusicImageDataset(image_root, midi_root, image_transform, max_seq_len=max_seq_len, max_midi_files=50)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    model = CNNRNNModel(input_channels=1, hidden_dim=1024, output_dim=3, max_seq_len=max_seq_len)

    learning_data = train_model(model, dataloader, epochs=10, device=device, learning_rate=0.0002)

    generate_chart(learning_data)

    generate_midi_from_selected_image(model, dataset, selected_image_path, "output_selected_from_main.mid")