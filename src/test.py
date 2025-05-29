import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import mido

from src.cnnrnn_model import CNNRNNModel


def sequence_to_midi(sequence, output_midi_path):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for pitch, duration in sequence:
        if pitch > 0:  # Ignore padding
            pitch = max(0, min(127, int(pitch)))
            duration = max(0, int(duration))
            track.append(mido.Message('note_on', note=pitch, velocity=64, time=0))
            track.append(mido.Message('note_off', note=pitch, velocity=0, time=duration))

    mid.save(output_midi_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model
model = CNNRNNModel(input_channels=3, hidden_dim=512, output_dim=2)
model.to(device)

# Load the saved model
model.load_state_dict(torch.load("model_mini.pth", map_location=device, weights_only=True))
model.eval()

# Image transformation (must match training)
image_transform = transforms.Compose([
    transforms.Resize((992, 1402)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess a single image
image_path = "data/images/mozart/mz_311_1/mz_311_1-1.png"  # Replace with your image path
image = Image.open(image_path).convert('RGB')
image = image_transform(image).unsqueeze(0)
image = image.to(device)

# Generate MIDI
with torch.no_grad():
    output = model(image)  # No target during inference
    predicted_sequence = output[0].cpu().detach().numpy().tolist()
    print(predicted_sequence)
    # predicted_sequence = [(int(abs(x) * 1000000), int(abs(y) * 1000000)) for x, y in predicted_sequence]
    # predicted_sequence = [(int(abs(x) * 500), int(abs(y) * 8)) for x, y in predicted_sequence]
    predicted_sequence = [(int(abs(x) * 127), int(abs(y) * 400)) for x, y in predicted_sequence]
    # predicted_sequence = [(int(abs(x)), int(abs(y))) for x, y in predicted_sequence]
    # predicted_sequence = [(int(p * 127 + 0.5), int(d * 10)) for p, d in predicted_sequence]
    print(predicted_sequence)
    sequence_to_midi(predicted_sequence, "generated_mozart.mid")


