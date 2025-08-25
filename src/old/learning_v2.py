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

        # Step 2: Randomly select up to 100 MIDI files
        random.shuffle(midi_files)
        self.selected_midi_files = midi_files[:max_midi_files]

        # Step 3: Collect all images corresponding to the selected MIDI files
        self.image_paths = []
        self.midi_features = {}

        for folder, author, midi_file in self.selected_midi_files:
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

# CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_channels=3, max_seq_len=100, output_dim=2):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        # Fully connected layers to output MIDI sequence
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, max_seq_len * output_dim),  # Output: [max_seq_len, output_dim]
            nn.ReLU()  # Ensure non-negative outputs for pitch and duration
        )

    def forward(self, x):
        # CNN feature extraction
        batch_size = x.size(0)
        features = self.conv_layers(x)  # Shape: [batch_size, 512, 1, 1]
        features = features.view(batch_size, -1)  # Flatten: [batch_size, 512]
        # Fully connected layers
        output = self.fc_layers(features)  # Shape: [batch_size, max_seq_len * output_dim]
        output = output.view(batch_size, 100, 2)  # Reshape: [batch_size, max_seq_len, output_dim]
        return output

# Training Loop
def train_model(model, dataloader, epochs=3, device=device):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjusted learning rate

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, midi_batch) in enumerate(dataloader):
            # Move data to GPU
            images = images.to(device)
            midi_batch = midi_batch.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, midi_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

        print(f"Epoch {epoch+1} completed, Average Loss: {running_loss / len(dataloader):.4f}")

# Generate MIDI
def sequence_to_midi(sequence, output_midi_path):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for pitch, duration in sequence:
        if pitch > 0:  # Ignore padding
            track.append(mido.Message('note_on', note=int(pitch), velocity=64, time=0))
            track.append(mido.Message('note_off', note=int(pitch), velocity=0, time=int(duration)))

    mid.save(output_midi_path)

# Example usage
image_root = "data/images"
midi_root = "data/processed_midi"

# Data transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloader
dataset = MusicImageDataset(image_root, midi_root, image_transform, max_seq_len=100, max_midi_files=100)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize and train model
model = CNNModel(input_channels=3, max_seq_len=100, output_dim=2)
train_model(model, dataloader, epochs=3, device=device)

# Generate MIDI from the first batch
model.eval()
with torch.no_grad():
    for images, midi_batch in dataloader:
        images = images.to(device)
        output = model(images)
        predicted_sequence = output[0].cpu().detach().numpy().tolist()
        sequence_to_midi(predicted_sequence, "output.mid")
        break