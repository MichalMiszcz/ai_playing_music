import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import mido

from src.cnnrnn_model_3_bitmap import CNNRNNModel


def sequence_to_midi(sequence, output_midi_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    left_track = mido.MidiTrack()
    right_track = mido.MidiTrack()

    current_time = 0
    events = []
    for hand, note, velocity, delta_time in sequence:
        current_time += delta_time
        events.append((current_time, hand, note, velocity))

    left_events = [e for e in events if e[1] == 0]
    right_events = [e for e in events if e[1] == 1]

    left_events.sort(key=lambda x: x[0])
    right_events.sort(key=lambda x: x[0])

    def add_events_to_track(track, events):
        prev_time = 0
        for time, _, note, velocity in events:
            delta_time = time - prev_time
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta_time))
            prev_time = time

    add_events_to_track(left_track, left_events)
    add_events_to_track(right_track, right_events)

    mid.tracks.append(left_track)
    mid.tracks.append(right_track)

    mid.save(output_midi_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CNNRNNModel(input_channels=1, hidden_dim=128, output_dim=4)
model.to(device)

# model.load_state_dict(torch.load("../data/models/model_bitmap_512notes200epochs.pth", map_location=device, weights_only=True))
model.load_state_dict(torch.load("../model_mini.pth", map_location=device, weights_only=True))
model.eval()

image_transform = transforms.Compose([
    transforms.Resize((1488, 450)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

image_path = "../my_images/my_midi_images/my_midi_files/song_1/song_1-1.png"
# image_path = "../my_images_my_simple_songs/my_midi_images/my_midi_files/c/c-1.png"
image = Image.open(image_path).convert('1')
image = image_transform(image).unsqueeze(0)
image = image.to(device)

with torch.no_grad():
    output = model(image)  # No target during inference
    predicted_sequence = output[0].cpu().detach().numpy().tolist()
    print(predicted_sequence)
    # predicted_sequence = [(int(abs(x) * 1000000), int(abs(y) * 1000000)) for x, y in predicted_sequence]
    # predicted_sequence = [(int(abs(x) * 500), int(abs(y) * 8)) for x, y in predicted_sequence]
    predicted_sequence = [(1 if h > 0.5 else 0, int(n * 127 + 0.5), int(v * 127 + 0.5), int(dt * 512 + 0.5))
            for h, n, v, dt in predicted_sequence]
    # predicted_sequence = [(int(abs(x)), int(abs(y))) for x, y in predicted_sequence]
    # predicted_sequence = [(int(p * 127 + 0.5), int(d * 10)) for p, d in predicted_sequence]
    print(predicted_sequence)
    sequence_to_midi(predicted_sequence, "../generated_simple_1.mid")


