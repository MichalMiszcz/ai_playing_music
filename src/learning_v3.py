import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import mido
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset
class MusicImageDataset(Dataset):
    def __init__(self, image_root, midi_root, image_transform=None, max_seq_len=100, max_midi_files=100):
        self.image_root = image_root
        self.midi_root = midi_root
        self.image_transform = image_transform if image_transform else transforms.ToTensor()
        self.max_seq_len = max_seq_len

        # Collect MIDI files
        midi_files = []
        for root, dirs, files in os.walk(midi_root):
            folder = os.path.basename(os.path.dirname(root))
            author = os.path.basename(root)
            for file in files:
                if file.endswith('.mid'):
                    midi_files.append((folder, author, os.path.join(root, file)))

        # Randomly select up to max_midi_files MIDI files
        random.shuffle(midi_files)
        self.selected_midi_files = midi_files[:max_midi_files]

        # Find max_duration across all selected MIDI files
        self.max_duration = 0
        for folder, author, midi_file in self.selected_midi_files:
            midi_seq = extract_notes_from_midi(midi_file)
            for _, duration in midi_seq:
                if duration > self.max_duration:
                    self.max_duration = duration
        if self.max_duration == 0:
            self.max_duration = 1920  # Default to a whole note if no durations found

        # Collect images and MIDI features
        self.image_paths = []
        self.midi_features = {}

        for folder, author, midi_file in self.selected_midi_files:
            midi_key = f"{folder}/{author}"
            midi_seq = extract_notes_from_midi(midi_file)
            # Normalize MIDI sequence
            midi_seq = [(note / 127.0, duration / self.max_duration) for note, duration in midi_seq]
            # Pad or truncate to max_seq_len
            if len(midi_seq) > self.max_seq_len:
                midi_seq = midi_seq[:self.max_seq_len]
            else:
                midi_seq += [(0.0, 0.0)] * (self.max_seq_len - len(midi_seq))
            self.midi_features[midi_key] = midi_seq

            file = os.path.splitext(os.path.basename(midi_file))[0]
            image_dir = os.path.join(image_root, author, file)
            if os.path.exists(image_dir):
                image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
                for img_file in image_files:
                    self.image_paths.append(os.path.join(image_dir, img_file))

        self.image_paths.sort()
        print(f"Selected {len(self.selected_midi_files)} MIDI files and {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        rel_path = os.path.relpath(img_path, self.image_root)
        composer, piece, _ = rel_path.split(os.sep)
        midi_key = f"{composer}/{piece}"

        image = Image.open(img_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)

        midi_seq = self.midi_features.get(midi_key, [(0.0, 0.0)] * self.max_seq_len)
        midi_tensor = torch.tensor(midi_seq, dtype=torch.float32)

        return image, midi_tensor

# Extract notes from MIDI (assuming durations in ticks)
def extract_notes_from_midi(midi_path):
    mid = mido.MidiFile(midi_path)
    notes = []
    current_time = 0
    note_starts = {}

    for msg in mid:
        current_time += msg.time  # Note: msg.time is in seconds; conversion needed if ticks desired
        if msg.type == 'note_on' and msg.velocity > 0:
            note_starts[msg.note] = current_time
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in note_starts:
                start_time = note_starts.pop(msg.note)
                duration = current_time - start_time
                notes.append((msg.note, duration))

    # Convert durations from seconds to ticks (simplified; assumes tempo and ticks_per_beat)
    ticks_per_beat = mid.ticks_per_beat if hasattr(mid, 'ticks_per_beat') else 480
    tempo = 500000  # Default tempo: 120 BPM (500000 microseconds per beat)
    for i, (note, duration) in enumerate(notes):
        # Convert seconds to ticks: (duration * ticks_per_beat) / (tempo / 1000000)
        duration_in_ticks = int((duration * ticks_per_beat * 1000000) / tempo)
        notes[i] = (note, duration_in_ticks)

    return notes

# CNN-RNN Model with sigmoid output
class CNNRNNModel(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=512, output_dim=2, rnn_layers=2):
        super(CNNRNNModel, self).__init__()
        # CNN for feature extraction
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Identity()
        self.cnn.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # RNN for sequence generation
        self.rnn = nn.LSTM(input_size=output_dim, hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

        # Projection for initial hidden state
        self.proj_h = nn.Linear(hidden_dim, hidden_dim * rnn_layers)
        self.proj_c = nn.Linear(hidden_dim, hidden_dim * rnn_layers)

    def forward(self, x, target=None):
        batch_size = x.size(0)
        features = self.cnn(x).view(batch_size, -1)

        # Initialize RNN hidden state
        h0 = self.proj_h(features).view(batch_size, self.rnn.num_layers, -1).transpose(0, 1).contiguous()
        c0 = self.proj_c(features).view(batch_size, self.rnn.num_layers, -1).transpose(0, 1).contiguous()

        if target is not None:
            # Teacher forcing during training
            input_seq = torch.cat([torch.zeros(batch_size, 1, 2).to(x.device), target[:, :-1, :]], dim=1)
            output, _ = self.rnn(input_seq, (h0, c0))
            output = self.linear(output)
            output = torch.sigmoid(output)  # Constrain outputs to [0,1]
            return output
        else:
            # Autoregressive generation during inference
            output_seq = []
            input_note = torch.zeros(batch_size, 1, 2).to(x.device)
            hidden = (h0, c0)
            for _ in range(100):  # Fixed sequence length
                output, hidden = self.rnn(input_note, hidden)
                output = self.linear(output)
                output = torch.sigmoid(output)
                output_seq.append(output)
                input_note = output
            return torch.cat(output_seq, dim=1)

# Training function with updated hyperparameters
def train_model(model, dataloader, epochs=50, device=device):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Reduced learning rate

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (images, midi_batch) in enumerate(dataloader):
            images = images.to(device)
            midi_batch = midi_batch.to(device)
            optimizer.zero_grad()
            output = model(images, midi_batch)
            loss = criterion(output, midi_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.6f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as 'model.pth'")

# Generate MIDI from image with scaling back to MIDI ranges
def generate_midi_from_selected_image(model, dataset, image_path, output_midi_path, device=device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    transform = image_transform
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        predicted_sequence = output[0].cpu().detach().numpy().tolist()
        # Scale back to MIDI ranges
        predicted_sequence = [(int(p * 127 + 0.5), int(d * dataset.max_duration + 0.5)) for p, d in predicted_sequence]
        sequence_to_midi(predicted_sequence, output_midi_path)

# Convert sequence to MIDI with durations in ticks
def sequence_to_midi(sequence, output_midi_path):
    mid = mido.MidiFile(ticks_per_beat=480)  # Standard ticks_per_beat
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for pitch, duration in sequence:
        if pitch > 0:  # Skip padding notes
            pitch = max(0, min(127, pitch))  # Clamp pitch to valid MIDI range
            duration = max(0, duration)  # Ensure non-negative duration
            track.append(mido.Message('note_on', note=pitch, velocity=64, time=0))
            track.append(mido.Message('note_off', note=pitch, velocity=0, time=duration))

    mid.save(output_midi_path)

# Image transformation
image_transform = transforms.Compose([
    transforms.Resize((496, 701)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Main execution
if __name__ == "__main__":
    # Paths (adjust as needed)
    image_root = "path/to/images"  # Replace with your image directory
    midi_root = "path/to/midi"     # Replace with your MIDI directory

    # Initialize dataset and dataloader
    dataset = MusicImageDataset(image_root, midi_root, image_transform, max_seq_len=100, max_midi_files=200)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize and train model
    model = CNNRNNModel(input_channels=3, hidden_dim=512, output_dim=2)
    train_model(model, dataloader, epochs=50, device=device)

    # Generate MIDI from a selected image
    selected_image_path = "path/to/selected/image.png"  # Replace with your image path
    generate_midi_from_selected_image(model, dataset, selected_image_path, "output_selected.mid")