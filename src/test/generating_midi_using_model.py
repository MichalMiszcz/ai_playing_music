import torch
from torchvision import transforms
from PIL import Image
import mido

from src.music_program.cnnrnn_model_4_greyscale import CNNRNNModel
from src.music_program.global_variables import *

model_path = "src/model_multi_lines_v11.pth"
# image_path = "src/all_data/generated/my_images_test/my_midi_images/my_midi_files/kotek/kotek-1.png"
# image_path = "src/all_data/generated/my_complex_images/my_midi_images/my_midi_files/song_1/song_1-1.png"
image_path = "src/kotek/kotek-1.png"
# image_path = "src/kotek_hr.png"
# image_path = "src/all_data/data_to_analyze/hi_res/song_7/song_7-1.png"
output_path = "src/output_midi.mid"

models_hidden_dim = 256
models_rnn_layers = 5


def sequence_to_midi(sequence, output_midi_path):
    mid = mido.MidiFile(ticks_per_beat=10080)
    right_track = mido.MidiTrack()

    current_time = 0
    events = []

    for note, velocity, delta_time in sequence:
        current_time += delta_time
        events.append((current_time, note, velocity))

    right_events = events

    right_events.sort(key=lambda x: x[0])

    def add_events_to_track(track, events):
        prev_time = 0
        for time, note, velocity in events:
            delta_time = time - prev_time
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta_time))
            prev_time = time

    add_events_to_track(right_track, right_events)
    mid.tracks.append(right_track)

    mid.save(output_midi_path)

    print("==== MIDI FILE INFO ====")
    print(f"Type: {mid.type}")
    print(f"Number of Tracks: {len(mid.tracks)}")
    print(f"Ticks per Beat: {mid.ticks_per_beat}")
    print(f"Length (seconds): {mid.length:.2f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CNNRNNModel(input_channels=1, hidden_dim=models_hidden_dim, output_dim=3, rnn_layers=models_rnn_layers)
model.to(device)

# loading model
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

image_transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

image = Image.open(image_path).convert('L')
image = image_transform(image).unsqueeze(0)
image = image.to(device)

with torch.no_grad():
    output = model(image)
    predicted_sequence = output[0].cpu().detach().numpy().tolist()
    predicted_sequence = predicted_sequence[:96]
    print(f"Raw predicted sequence: {predicted_sequence}")

    final_predicted_sequence = []
    for norm_note, norm_vel, norm_dt in predicted_sequence:
        note_idx = int(norm_note * (NUM_NOTES - 1.0) + 0.5)
        note_idx = max(0, min(note_idx, NUM_NOTES - 1))
        midi_note = WHITE_KEYS_MIDI[note_idx]

        velocity_idx = int(norm_vel * (NUM_VELOCITIES - 1.0) + 0.5)
        velocity_idx = max(0, min(velocity_idx, NUM_VELOCITIES - 1))
        velocity = VELOCITY[velocity_idx]

        delta_time_idx = int(norm_dt * (NUM_DELTA_TIME - 1.0) + 0.5)
        delta_time_idx = max(0, min(delta_time_idx, NUM_DELTA_TIME - 1.0))
        delta_time = int(DELTA_TIME[delta_time_idx])

        final_predicted_sequence.append((midi_note, velocity, delta_time))

    final_predicted_sequence = final_predicted_sequence[:96]
    print(f"Final predicted sequence: {final_predicted_sequence}")

    sequence_to_midi(final_predicted_sequence, output_path)


