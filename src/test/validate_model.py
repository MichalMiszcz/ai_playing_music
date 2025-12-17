import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

from src.music_program.cnnrnn_model_4_greyscale import CNNRNNModel
from src.music_program.global_variables import *
from src.music_program.music_image_dataset_4_greyscale import MusicImageDataset
from src.test.accuracy import *

model_path = "model_multi_lines.pth"
image_root_test = "all_data/generated/my_complex_images/my_midi_images"
midi_root_test = "all_data/generated/generated_complex_midi_processed"

midi_columns = ['midi_note', 'velocity', 'delta_time']

# model_path = "model_new_bigeye.pth"
# image_root_test = "all_data/generated/my_images_test_q/my_midi_images"
# midi_root_test = "all_data/generated/generated_songs_processed_test_q"

max_seq_len = 96
max_midi_files = 128
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
model = CNNRNNModel(input_channels=1, hidden_dim=512, output_dim=3, rnn_layers=6, max_seq_len=max_seq_len)
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

def calculate_metric_for_column(df_predicted, df_source, column: str):
    notes_stats_df = pd.DataFrame()
    dtw_score = dynamic_time_warping_score(df_predicted, df_source, column)
    notes_stats_df['dtw_score'] = [dtw_score]
    return notes_stats_df

def validate_predicted_midi(df_predicted: pd.DataFrame, df_source: pd.DataFrame):
    stats_df = pd.DataFrame()

    stats = dynamic_time_warping_score_multi_col(df_predicted, df_source, [midi_columns[0], 'time'])

    stats_df['DTW score'] = [stats]

    return stats_df
    # return stats['midi_note'], stats['velocity'], stats['delta_time']

def midi_to_df(midi_seq):
    df_midi = pd.DataFrame(midi_seq, columns=midi_columns)
    df_midi['delta_time_s'] = df_midi['delta_time'] / (10080 * 2)  # quarter note = 0.5s
    df_midi['time'] = df_midi['delta_time_s'].cumsum()

    return df_midi

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
            predicted_sequence = predicted_sequence[:max_seq_len]

            midi_batch = midi_batch.cpu()
            midi_batch = midi_batch.tolist()
            midi_batch = midi_batch[0]

            predicted_midi = from_raw_to_midi(predicted_sequence)
            source_midi = from_raw_to_midi(midi_batch)

            df_predicted_midi = midi_to_df(predicted_midi)
            df_source_midi = midi_to_df(source_midi)

            df_tmp_notes = validate_predicted_midi(df_predicted_midi, df_source_midi)
            df_notes = pd.concat([df_notes, df_tmp_notes], ignore_index=True)
            # df_velocity = pd.concat([df_velocity, df_tmp_velocity], ignore_index=True)
            # df_delta_time = pd.concat([df_delta_time, df_tmp_delta_time], ignore_index=True)

            # print("Predicted:   ", predicted_midi)
            # print("Source:      ", source_midi)

            # print("")

    df_notes.to_csv("csv/notes_stats.csv", index=False)
    # df_velocity.to_csv("csv/velocity_stats.csv", index=False)
    # df_delta_time.to_csv("csv/delta_time_stats.csv", index=False)

if __name__ == '__main__':
    main()