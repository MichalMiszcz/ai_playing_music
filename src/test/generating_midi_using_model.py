import torch
from torchvision import transforms
from PIL import Image
import mido

from src.music_program.cnnrnn_model_4_greyscale import CNNRNNModel
from src.music_program.global_variables import NUM_NOTES, WHITE_KEYS_MIDI

model_path = "model_multi_notes.pth"
image_path = "all_data/generated/my_images/my_midi_images/my_midi_files/song_1/song_1-1.png"
output_path = "output_midi.mid"


def sequence_to_midi(sequence, output_midi_path):
    mid = mido.MidiFile(ticks_per_beat=480)
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

model = CNNRNNModel(input_channels=1, hidden_dim=512, output_dim=3, rnn_layers=4)
model.to(device)

# loading model
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

image_transform = transforms.Compose([
    # transforms.Resize((SIZE_X, SIZE_Y)), # Resizing image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

image = Image.open(image_path).convert('L')
image = image_transform(image).unsqueeze(0)
image = image.to(device)

with torch.no_grad():
    output = model(image)
    predicted_sequence = output[0].cpu().detach().numpy().tolist()
    predicted_sequence = predicted_sequence[:32]
    print(f"Raw predicted sequence: {predicted_sequence}")

    final_predicted_sequence = []
    for norm_note, norm_vel, norm_dt in predicted_sequence:
        note_idx = int(norm_note * (NUM_NOTES - 1.0) + 0.5)
        note_idx = max(0, min(note_idx, NUM_NOTES - 1))
        midi_note = WHITE_KEYS_MIDI[note_idx]

        velocity = int(norm_vel * 127.0 + 0.5)
        delta_time = int(norm_dt * 1008 + 0.5)

        print(norm_dt)
        print(delta_time)

        final_predicted_sequence.append((midi_note, velocity, delta_time))

    final_predicted_sequence = final_predicted_sequence[:32]
    print(f"Final predicted sequence: {final_predicted_sequence}")

    sequence_to_midi(final_predicted_sequence, output_path)


