import os

import mido
import torch
from torchvision.transforms import v2

from src.music_program.utils.global_variables import *
from src.music_program.model.cnn_model_v11_val import MusicModel
from src.music_program.utils.yolo_cutting_image import segment_image
from src.music_program.utils.sequence_modifications import time_series_to_midi_seq

note_to_index = {midi_num: i for i, midi_num in enumerate(WHITE_KEYS_MIDI)}
velocity_to_index = {midi_num: i for i, midi_num in enumerate(VELOCITY)}
delta_time_to_index = {midi_num: i for i, midi_num in enumerate(DELTA_TIME)}


# image_folder = "src/all_data/generated/my_complex_images_test/my_midi_images/my_midi_files/song_1"
image_folder = "src/all_data/other"
generated_midi_folder = "generated_midi/model_v900_03"
# file_name = "song_1-1"
file_name = "billie"
file_type = ".png"

midi_columns = ['midi_note', 'velocity', 'delta_time']


version = 900
subversion = '03'

max_seq_len = 96
max_series_len = 32

batch_size = 1
features_number = 32
hidden_dim = 64


version_name = str(version) + '_' + str(subversion) if subversion is not None else str(version)
print(f'Version name: {version_name}')

model_path = f"src/_models/image_to_midi/model_best_v{version_name}_checkpoint.pth"

model_mode = "my_model"
res = "low"
csv_file_name = f"src/csv/notes_stats_{model_mode}_{res}_index{version_name}.csv"


left_hand_tracks = ['Piano left', 'Left']
right_hand_tracks = ['Piano right', 'Right', 'Track 0']

image_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    print(f"\nMIDI saved as {output_midi_path}\n")


    print("==== MIDI FILE INFO ====")
    print(f"Type: {mid.type}")
    print(f"Number of Tracks: {len(mid.tracks)}")
    print(f"Ticks per Beat: {mid.ticks_per_beat}")
    print(f"Length (seconds): {mid.length:.2f}")


def generate(image):
    model = MusicModel(features_number, hidden_dim, max_series_len)
    model.to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        staff_list = segment_image(path_to_image=image)

        predicted_seq = []
        for _, (_, staff) in enumerate(staff_list):
            staff = staff.convert('L')
            staff = image_transform(staff)
            staff = staff.to(device)
            staff = staff.unsqueeze(0)

            output_notes, output_times = model(staff)
            predicted_notes = output_notes[0].cpu().detach().numpy().tolist()
            predicted_times = output_times[0].cpu().detach().numpy().tolist()
            predicted_notes = predicted_notes[:max_series_len]
            predicted_times = predicted_times[:max_series_len]
            notes = [note_logit.index(max(note_logit)) for note_logit in predicted_notes]
            times = [note_logit.index(max(note_logit)) for note_logit in predicted_times]
            predicted = list(map(list, zip(notes, times)))

            predicted = [[note, time] for note, time in predicted if note != 8]
            predicted_seq.extend(predicted)

        if len(predicted_seq) < max_seq_len:
            predicted_seq.extend([[0, 0]] * (max_seq_len - len(predicted_seq)))

        predicted_midi_seq = time_series_to_midi_seq(predicted_seq, mode="new")

    return predicted_midi_seq


if __name__ == '__main__':
    img_file = file_name + file_type
    image_path = os.path.join(image_folder, img_file)

    midi_file = file_name + '.midi'
    midi_path = os.path.join(generated_midi_folder, midi_file)

    midi_seq = generate(image_path)
    sequence_to_midi(midi_seq, midi_path)