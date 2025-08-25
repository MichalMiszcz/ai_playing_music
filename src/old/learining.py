import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import mido
import random

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset
class MusicImageDataset(Dataset):
    def __init__(self, image_root, midi_root, image_transform=None, max_seq_len=100, max_midi_files=100):
        self.image_root = image_root
        self.midi_root = midi_root
        self.image_transform = image_transform if image_transform else transforms.ToTensor()
        self.max_seq_len = max_seq_len

        # Step 1: Collect all MIDI files
        midi_files = []
        for root, dirs, files in os.walk(midi_root):
            folder = os.path.basename(os.path.dirname(root))  # e.g., "albeniz"
            author = os.path.basename(root)  # e.g., "alb_esp1"
            for file in files:
                if file.endswith('.mid'):
                    midi_files.append((folder, author, os.path.join(root, file)))
        # print(f"Found {len(midi_files)} MIDI files in {midi_root}")

        # Step 2: Randomly select up to 100 MIDI files
        random.shuffle(midi_files)
        self.selected_midi_files = midi_files[:max_midi_files]

        # Step 3: Collect all images corresponding to the selected MIDI files
        self.image_paths = []
        self.midi_features = {}

        for folder, author, midi_file in self.selected_midi_files:
            # print(f"Processing MIDI: {folder}/{author} at {midi_file}")
            # Load MIDI features
            midi_key = f"{folder}/{author}"
            midi_seq = extract_notes_from_midi(midi_file)
            # Pad or truncate MIDI sequence to max_seq_len
            if len(midi_seq) > self.max_seq_len:
                midi_seq = midi_seq[:self.max_seq_len]
            else:
                midi_seq += [(0, 0)] * (self.max_seq_len - len(midi_seq))
            self.midi_features[midi_key] = midi_seq

            # Collect all images for this piece
            file = os.path.splitext(os.path.basename(midi_file))[0]
            image_dir = os.path.join(image_root, author, file)
            if os.path.exists(image_dir):
                image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
                for file in image_files:
                    self.image_paths.append(os.path.join(image_dir, file))

        self.image_paths.sort()  # Ensure consistent order
        print(f"Selected {len(self.selected_midi_files)} MIDI files and {len(self.image_paths)} images.")

        # Raise an error if no images are found
        if len(self.image_paths) == 0:
            raise ValueError("No images found for the selected MIDI files. Check directory paths and file structure.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Extract composer and piece from the path
        rel_path = os.path.relpath(img_path, self.image_root)
        composer, piece, _ = rel_path.split(os.sep)
        midi_key = f"{composer}/{piece}"

        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)

        # Get corresponding MIDI sequence
        midi_seq = self.midi_features.get(midi_key, [(0, 0)] * self.max_seq_len)
        midi_tensor = torch.tensor(midi_seq, dtype=torch.float32)

        return image, midi_tensor

# Feature extraction from MIDI
def extract_notes_from_midi(midi_path):
    mid = mido.MidiFile(midi_path)
    notes = []
    current_time = 0
    note_starts = {}

    for msg in mid:
        current_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            note_starts[msg.note] = current_time
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in note_starts:
                start_time = note_starts.pop(msg.note)
                duration = current_time - start_time
                notes.append((msg.note, duration))

    return notes

# CNN-Transformer Model
class CNNTransformerModel(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=512, output_dim=2, nhead=2, num_layers=2):
        super(CNNTransformerModel, self).__init__()
        # CNN for feature extraction from images
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Identity()  # Remove the final fully connected layer
        self.cnn.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

        # Transformer for sequence generation
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=512,
                batch_first=True),
            num_layers=num_layers
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # CNN feature extraction
        batch_size = x.size(0)
        features = self.cnn(x)  # Shape: [batch_size, hidden_dim, 1, 1]
        features = features.view(batch_size, -1)  # Flatten: [batch_size, hidden_dim]

        # Prepare for Transformer (add sequence dimension)
        seq_len = 100  # Fixed sequence length (same as max_seq_len in dataset)
        features = features.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim]

        # Transformer encoding
        output = self.transformer(features)
        output = self.linear(output)
        return output

# Training Loop
def train_model(model, dataloader, epochs=3, device=device):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    i = 0
    for epoch in range(epochs):
        i += 1
        model.train()
        j = 0
        for images, midi_batch in dataloader:
            j += 1
            print(f"Training {i}, {j}")
            # Move data to GPU
            images = images.to(device)
            midi_batch = midi_batch.to(device)

            optimizer.zero_grad()
            output = model(images)
            # Ensure target matches output shape
            loss = criterion(output, midi_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Example usage
image_root = "data/images"
midi_root = "data/processed_midi"

# Data transformation
image_transform = transforms.Compose([
    transforms.Resize((496, 701)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloader
dataset = MusicImageDataset(image_root, midi_root, image_transform, max_seq_len=100, max_midi_files=100)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize and train model
model = CNNTransformerModel(input_channels=3, hidden_dim=512, output_dim=2)
train_model(model, dataloader, device=device)

# Generate MIDI (placeholder) with GPU support
def sequence_to_midi(sequence, output_midi_path):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for pitch, duration in sequence:
        if pitch > 0:  # Ignore padding
            track.append(mido.Message('note_on', note=int(pitch), velocity=64, time=0))
            track.append(mido.Message('note_off', note=int(pitch), velocity=0, time=int(duration)))

    mid.save(output_midi_path)

# Example: Generate a MIDI file from the first batch
model.eval()
with torch.no_grad():
    for images, midi_batch in dataloader:
        images = images.to(device)
        output = model(images)
        predicted_sequence = output[0].cpu().detach().numpy().tolist()
        sequence_to_midi(predicted_sequence, "output.mid")
        break
