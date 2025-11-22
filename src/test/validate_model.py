import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

from src.music_program.cnnrnn_model_4_greyscale import CNNRNNModel
from src.music_program.global_variables import *
from src.music_program.music_image_dataset_4_greyscale import MusicImageDataset
from src.test.accuracy import *

model_path = "model_multi_notes_v5.pth"
image_root_test = "all_data/generated/my_images_test/my_midi_images"
midi_root_test = "all_data/generated/generated_songs_processed_test"

max_seq_len = 32
max_midi_files = 4
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
model = CNNRNNModel(input_channels=1, hidden_dim=256, output_dim=3, rnn_layers=5)
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

    final_predicted_sequence = final_predicted_sequence[:32]
    return final_predicted_sequence

def store_stats(predicted, source, max_len, stats_name):
    mae = mean_absolute_error(predicted, source, max_len)
    mse = mean_square_error(predicted, source, max_len)
    rmse = root_mean_square_error(predicted, source, max_len)
    er = show_errors(predicted, source, max_len)
    pc = percent_correct(predicted, source, max_len)

    print(f"--- {stats_name} ---")
    print("Mean absolute error: ", mae)
    print("Mean square error: ", mse)
    print("Root mean square error: ", rmse)
    print("Indexes of errors: ", er)
    print("Percent of correct: ", pc, "%")
    print()

    results = {'mean_absolute_error': mae, 'mean_square_error': mse, 'root_mean_square_error': rmse, 'indexes_of_errors': [er], 'percent_correct': pc}
    df_temp = pd.DataFrame(results)
    return df_temp

def validate_predicted_midi(predicted: list, source: list):
    notes_predicted = [n for n, _, _ in predicted]
    notes_source = [n for n, _, _ in source]
    velocity_predicted = [v for _, v, _ in predicted]
    velocity_source = [v for _, v, _ in source]
    delta_time_predicted = [dt for _, _, dt in predicted]
    delta_time_source = [dt for _, _, dt in source]

    df_tmp_notes = store_stats(notes_predicted, notes_source, max_seq_len, stats_name='NOTES')
    df_tmp_velocity = store_stats(velocity_predicted, velocity_source, max_seq_len, stats_name='VELOCITY')
    df_tmp_delta_time = store_stats(delta_time_predicted, delta_time_source, max_seq_len, stats_name='DELTA TIME')

    return df_tmp_notes, df_tmp_velocity, df_tmp_delta_time

def main():
    df_notes = pd.DataFrame()
    df_velocity = pd.DataFrame()
    df_delta_time = pd.DataFrame()

    for i, (images, midi_batch) in enumerate(val_dataloader):
        with torch.no_grad():
            images = images.to(device)
            midi_batch = midi_batch.to(device)

            output = model(images, midi_batch)
            predicted_sequence = output[0].cpu().detach().numpy().tolist()
            predicted_sequence = predicted_sequence[:32]

            midi_batch = midi_batch.cpu()
            midi_batch = midi_batch.tolist()
            midi_batch = midi_batch[0]

            predicted_midi = from_raw_to_midi(predicted_sequence)
            source_midi = from_raw_to_midi(midi_batch)

            print("Number of note sheets: ", max_midi_files)
            df_tmp_notes, df_tmp_velocity, df_tmp_delta_time = validate_predicted_midi(predicted_midi, source_midi)
            df_notes = pd.concat([df_notes, df_tmp_notes], ignore_index=True)
            df_velocity = pd.concat([df_velocity, df_tmp_velocity], ignore_index=True)
            df_delta_time = pd.concat([df_delta_time, df_tmp_delta_time], ignore_index=True)

            print("Predicted:   ", predicted_midi)
            print("Source:      ", source_midi)

            print("")

    df_notes.to_csv("csv/notes_predicted.csv", index=False)
    df_velocity.to_csv("csv/velocity_predicted.csv", index=False)
    df_delta_time.to_csv("csv/delta_time_predicted.csv", index=False)

if __name__ == '__main__':
    main()