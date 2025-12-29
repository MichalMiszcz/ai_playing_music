import os

import mido
import torch
from sympy.codegen.cnodes import sizeof
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

from src.music_program.cnnrnn_model_4_greyscale import CNNRNNModel
from src.music_program.global_variables import *
from src.music_program.music_image_dataset_4_greyscale import MusicImageDataset
from src.test.accuracy import *
from src.music_program.global_variables import *

note_to_index = {midi_num: i for i, midi_num in enumerate(WHITE_KEYS_MIDI)}
velocity_to_index = {midi_num: i for i, midi_num in enumerate(VELOCITY)}
delta_time_to_index = {midi_num: i for i, midi_num in enumerate(DELTA_TIME)}

model_path = "model_multi_lines_v10.pth"
image_root_test = "all_data/generated/my_complex_images/my_midi_images"
midi_root_test = "all_data/generated/generated_complex_midi_processed"

midi_columns = ['midi_note', 'velocity', 'delta_time']

# model_path = "model_new_bigeye.pth"
# image_root_test = "all_data/generated/my_images_test_q/my_midi_images"
# midi_root_test = "all_data/generated/generated_songs_processed_test_q"

max_seq_len = 96
max_midi_files = 32
left_hand_tracks = ['Piano left', 'Left']
right_hand_tracks = ['Piano right', 'Right', 'Track 0']


image_transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset
val_dataset = MusicImageDataset(image_root_test, midi_root_test, left_hand_tracks, right_hand_tracks, image_transform, max_seq_len=max_seq_len, max_midi_files=max_midi_files)
val_dataloader = DataLoader(val_dataset, shuffle=False)

# Loading model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNRNNModel(input_channels=1, hidden_dim=256, output_dim=3, rnn_layers=4, max_seq_len=max_seq_len)
model.to(device)

model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

def from_raw_to_midi(sequence):
    final_predicted_sequence = []
    for norm_note, norm_vel, norm_dt in sequence:
        note_idx = int(norm_note * (NUM_NOTES - 1.0) + 0.5)
        note_idx = max(0, min(note_idx, NUM_NOTES - 1))
        midi_note = WHITE_KEYS_MIDI[note_idx]

        velocity_idx = int(norm_vel * (NUM_VELOCITIES - 1.0) + 0.5)
        velocity_idx = max(0, min(velocity_idx, NUM_VELOCITIES - 1))
        velocity = VELOCITY[velocity_idx]

        delta_time_idx = int(norm_dt * (NUM_DELTA_TIME - 1.0) + 0.5)
        delta_time_idx = max(0, min(delta_time_idx, NUM_DELTA_TIME - 1.0))
        delta_time = DELTA_TIME[delta_time_idx]

        final_predicted_sequence.append((midi_note, velocity, delta_time))

    final_predicted_sequence = final_predicted_sequence[:max_seq_len]
    return final_predicted_sequence

def calculate_metric_for_column(df_predicted, df_source, column: str):
    notes_stats_df = pd.DataFrame()
    dtw_score = dynamic_time_warping_score(df_predicted, df_source, column)
    notes_stats_df['dtw_score'] = [dtw_score]
    return notes_stats_df

def validate_predicted_midi(df_predicted: pd.DataFrame, df_source: pd.DataFrame):
    stats_df = pd.DataFrame()

    df_predicted['velocity_normalized'] = df_predicted[midi_columns[1]] / 90
    df_source['velocity_normalized'] = df_source[midi_columns[1]] / 90

    dtw_score = dynamic_time_warping_score_multi_col(df_predicted, df_source, [midi_columns[0], 'time'])
    levenstein = edit_distance_multi_col(df_predicted, df_source, [midi_columns[0], 'velocity_normalized', 'delta_time_s'])
    frechet = discrete_frechet(df_predicted, df_source, [midi_columns[0], 'velocity_normalized', 'time'])

    stats_df['DTW score'] = [dtw_score]
    stats_df['Levenstein score'] = [levenstein]
    stats_df['DFS score'] = [frechet]

    return stats_df
    # return stats['midi_note'], stats['velocity'], stats['delta_time']

def midi_to_df(midi_seq):
    df_midi = pd.DataFrame(midi_seq, columns=midi_columns)
    df_midi['delta_time_s'] = df_midi['delta_time'] / (10080 * 2)  # quarter note = 0.5s
    df_midi['time'] = df_midi['delta_time_s'].cumsum()

    return df_midi

def calculate_measures(predicted_sequence, source_sequence):
    df_predicted_midi = midi_to_df(predicted_sequence)
    df_source_midi = midi_to_df(source_sequence)

    df_results = validate_predicted_midi(df_predicted_midi, df_source_midi)

    print("Predicted:   ", predicted_sequence)
    print("Source:      ", source_sequence)

    return df_results


def main():
    df_final_results = pd.DataFrame()
    test_mode = "midi"

    right_hand_tracks_for_validation = ['Piano right', 'Right', 'Track 0', 'Track', 'Voice']
    midi_folder_path = "all_data/model_generated/audiveris/low_res"
    midi_source_folder_path = "all_data/model_generated/source_midi"

    if test_mode == "model":
        for i, (images, midi_batch) in enumerate(val_dataloader):
            with torch.no_grad():
                images = images.to(device)
                midi_batch = midi_batch.to(device)

                output = model(images, midi_batch)
                predicted_sequence = output[0].cpu().detach().numpy().tolist()
                predicted_sequence = predicted_sequence[:max_seq_len]

                midi_batch = midi_batch.cpu()
                midi_batch = midi_batch.tolist()
                midi_batch = midi_batch[0]

                predicted_midi_seq = from_raw_to_midi(predicted_sequence)
                source_midi_seq = from_raw_to_midi(midi_batch)

                df_results = calculate_measures(predicted_midi_seq, source_midi_seq)

                df_final_results = pd.concat([df_final_results, df_results], ignore_index=True)

    elif test_mode == "midi":
        def get_sequence_from_mido(mido: mido.MidiFile, mode = "midi"):
            sequence = []
            for j, track in enumerate(mido.tracks):
                if track.name in right_hand_tracks_for_validation:
                    for msg in track:
                        if msg.type in ('note_on', 'note_off'):
                            time = msg.time
                            velocity = msg.velocity if msg.type == 'note_on' else 0
                            if mode == "sheet_vision":
                                time = int(msg.time * 10.5)
                                velocity = int(velocity * 0.9)

                            sequence.append((msg.note, velocity, time))

            return sequence

        for root, _, files in os.walk(midi_folder_path):
            for i, file in enumerate(files):
                midi_dir = os.path.join(root, file)
                mid = mido.MidiFile(midi_dir)
                mid_source_dir = os.path.join(midi_source_folder_path, file)
                mid_source = mido.MidiFile(mid_source_dir)
                sequence = get_sequence_from_mido(mid, "audiveris")
                sequence_source = get_sequence_from_mido(mid_source)

                df_results = calculate_measures(sequence, sequence_source)
                df_final_results = pd.concat([df_final_results, df_results], ignore_index=True)

                print(sequence)
                print(sequence_source)


    df_final_results.to_csv("csv/notes_stats.csv", index=False)

if __name__ == '__main__':
    main()