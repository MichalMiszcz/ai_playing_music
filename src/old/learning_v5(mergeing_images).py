import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import mido
import random

image_root = "data/images"
midi_root = "data/processed_midi"
selected_image_path = "../data/images/albeniz/alb_esp1/alb_esp1-1.png"  # Replace with your image path

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

global count_tracks
count_tracks = 0

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
        records_to_remove = []

        for folder, author, midi_file in self.selected_midi_files:
            # print(f"Processing MIDI: {folder}/{author} at {midi_file}")
            # Load MIDI features
            midi_name = os.path.splitext(os.path.basename(midi_file))[0]
            midi_key = f"{author}/{midi_name}"
            try:
                midi_seq = extract_notes_from_midi(midi_file)
                # Pad or truncate MIDI sequence to max_seq_len
                if len(midi_seq) > self.max_seq_len:
                    midi_seq = midi_seq[:self.max_seq_len]
                else:
                    midi_seq += [(0, 0)] * (self.max_seq_len - len(midi_seq))
                self.midi_features[midi_key] = midi_seq
            except Exception as e:
                print(f"Error processing MIDI {midi_file}: {e}")
                records_to_remove.append((folder, author, midi_file))
                continue

            # Collect all images for this piece
            file = os.path.splitext(os.path.basename(midi_file))[0]
            image_dir = os.path.join(image_root, author, file)
            images_for_one_piece = []
            if os.path.exists(image_dir):
                image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
                for file in image_files:

                    images_for_one_piece.append(os.path.join(image_dir, file))
                self.image_paths.append(images_for_one_piece)

                print(self.image_paths)

        for record in records_to_remove:
            self.selected_midi_files.remove(record)

        # Compute max_duration for normalization
        all_durations = [duration for midi_seq in self.midi_features.values() for _, duration in midi_seq if duration > 0]
        self.max_duration = max(all_durations) if all_durations else 1  # Avoid division by zero

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
        print(img_path)
        rel_path = os.path.relpath(img_path, self.image_root)
        composer, piece, _ = rel_path.split(os.sep)
        midi_key = f"{composer}/{piece}"

        # Load image
        print(img_path)
        image = Image.open(img_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)

        # print("Midi key: ", midi_key)
        list_of_keys = self.midi_features.keys()
        # print("Midi features: ", list_of_keys)

        # Get corresponding MIDI sequence
        midi_seq = self.midi_features.get(midi_key, [(0, 0)] * self.max_seq_len)

        normalized_seq = [(note / 127.0, duration / self.max_duration) for note, duration in midi_seq]
        midi_tensor = torch.tensor(normalized_seq, dtype=torch.float32)

        return image, midi_tensor

def extract_notes_from_midi(midi_path):
    """
    Extracts notes from the "Piano right" and "Piano left" tracks of a MIDI file.

    Args:
        midi_path (str): Path to the MIDI file.

    Returns:
        list: A list of (note, duration) tuples sorted by start time if both tracks are present.
        None: If either "Piano right" or "Piano left" track is missing or if the file cannot be loaded.
    """
    # Load the MIDI file
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"Error loading MIDI {midi_path}: {e}")
        return None

    # Initialize variables to store the tracks
    right_hand_track = None
    left_hand_track = None

    # Search for "Piano right" and "Piano left" tracks
    for track in mid.tracks:
        if track.name == "Piano right":
            right_hand_track = track
        elif track.name == "Piano left":
            left_hand_track = track

    # Check if both tracks are present
    if right_hand_track is None or left_hand_track is None:
        print(f"MIDI file {midi_path} does not have both 'Piano right' and 'Piano left' tracks.")
        return None

    # Helper function to extract notes from a track
    def extract_notes_from_track(track):
        notes = []
        current_time = 0
        note_starts = {}
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_starts[msg.note] = current_time
            elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)):
                if msg.note in note_starts:
                    start_time = note_starts.pop(msg.note)
                    duration = current_time - start_time
                    notes.append((start_time, msg.note, duration))
        return notes

    # Extract notes from both tracks
    right_hand_notes = extract_notes_from_track(right_hand_track)
    left_hand_notes = extract_notes_from_track(left_hand_track)

    # Combine notes from both hands and sort by start time
    all_notes = right_hand_notes + left_hand_notes
    all_notes.sort(key=lambda x: x[0])  # Sort by start_time

    # Format the output as a list of (note, duration) tuples
    notes = [(note, duration) for _, note, duration in all_notes]

    # print(notes)

    return notes

# CNN-RNN Model
class CNNRNNModel(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=512, output_dim=2, rnn_layers=2, max_seq_len = 100):
        super(CNNRNNModel, self).__init__()
        self.max_seq_len = max_seq_len  # Match dataset
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Identity()
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
            input_seq = torch.cat([torch.zeros(batch_size, 1, 2).to(x.device), target[:, :-1, :]], dim=1)
            output, _ = self.rnn(input_seq, (h0, c0))
            output = self.linear(output)
            output = torch.sigmoid(output)
            return output
        else:
            output_seq = []
            input_note = torch.zeros(batch_size, 1, 2).to(x.device)
            hidden = (h0, c0)
            for _ in range(self.max_seq_len):
                output, hidden = self.rnn(input_note, hidden)
                output = self.linear(output)
                output = torch.sigmoid(output)
                output_seq.append(output)
                input_note = output
            return torch.cat(output_seq, dim=1)

# Training function
def train_model(model, dataloader, epochs=50, device=device, learning_rate=0.0005):
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
            # print(f"Sample output: {output[0, :5].cpu().detach().numpy()}")
            # print(f"Sample target: {midi_batch[0, :5].cpu().numpy()}")
            loss = criterion_mse(output, midi_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.6f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), '../model.pth')
    print("Model saved as 'model.pth'")

# Generate MIDI
def generate_midi_from_selected_image(model, dataset, image_path, output_midi_path, device=device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    transform = image_transform
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        predicted_sequence = output[0].cpu().detach().numpy().tolist()
        print(f"Raw predicted sequence: {predicted_sequence[:10]}")
        predicted_sequence = [(int(p * 127 + 0.5), int(d * dataset.max_duration + 0.5)) for p, d in predicted_sequence]
        print(f"Scaled predicted sequence: {predicted_sequence[:10]}")
        sequence_to_midi(predicted_sequence, output_midi_path)

# Convert sequence to MIDI
def sequence_to_midi(sequence, output_midi_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for pitch, duration in sequence:
        if pitch > 0:
            pitch = max(0, min(127, pitch))
            duration = max(0, duration)
            track.append(mido.Message('note_on', note=pitch, velocity=64, time=0))
            track.append(mido.Message('note_off', note=pitch, velocity=0, time=duration))

    mid.save(output_midi_path)

# Image transformation
image_transform = transforms.Compose([
    transforms.Resize((992, 1402)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Main execution
if __name__ == "__main__":
    # Initialize dataset and dataloader
    max_seq_len = 400
    dataset = MusicImageDataset(image_root, midi_root, image_transform, max_seq_len=max_seq_len, max_midi_files=15)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize and train model
    model = CNNRNNModel(input_channels=3, hidden_dim=512, output_dim=2, max_seq_len=max_seq_len)
    train_model(model, dataloader, epochs=40, device=device, learning_rate=0.025)

    # Generate MIDI
    generate_midi_from_selected_image(model, dataset, selected_image_path, "../output_selected_from_main.mid")