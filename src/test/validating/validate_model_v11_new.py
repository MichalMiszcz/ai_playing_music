import pandas as pd
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import to_pil_image
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from src.music_program.utils.global_variables import *
from src.music_program.model.cnn_model_v11_val import MusicModel
from src.music_program.dataset.music_image_dataset_11 import MusicImageDataset
from src.test.testing_scripts.testing_yolo_cutting_image import segment_image
from src.test.validating.counting_errors import count_errors
from src.test.validating.validate_model import time_series_to_midi_seq, calculate_measures

note_to_index = {midi_num: i for i, midi_num in enumerate(WHITE_KEYS_MIDI)}
velocity_to_index = {midi_num: i for i, midi_num in enumerate(VELOCITY)}
delta_time_to_index = {midi_num: i for i, midi_num in enumerate(DELTA_TIME)}

image_root_test = "src/all_data/generated/my_complex_images_test/my_midi_images"
midi_root_test = "src/all_data/generated/generated_complex_midi_processed_test"

midi_columns = ['midi_note', 'velocity', 'delta_time']


version = 900
subversion = '020'

max_seq_len = 96
max_series_len = 32

batch_size = 1
features_number = 32
hidden_dim = 256


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

val_dataset = MusicImageDataset(image_root_test, midi_root_test, left_hand_tracks, right_hand_tracks,
                                image_transform, max_seq_len=max_seq_len, max_series_len=48,
                                max_midi_files=80, modify_image=False, learning=False)

val_dataloader = DataLoader(val_dataset, shuffle=False, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate():
    df_final_results = pd.DataFrame()

    model = MusicModel(features_number, hidden_dim, max_series_len)

    model.to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_source_notes = []
    all_predicted_notes = []
    all_source_times = []
    all_predicted_times = []
    for i, (image, midi_batch) in enumerate(val_dataloader):
        with torch.no_grad():
            image = image.to(device)
            image = image.squeeze(dim=0)
            image = to_pil_image(image)
            # image.show()
            staff_list = segment_image(source_image=image)

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
                # predicted_seq = predicted_times[:max_seq_len]

            if len(predicted_seq) < max_seq_len:
                predicted_seq.extend([[0, 0]] * (max_seq_len - len(predicted_seq)))
            midi_seq = midi_batch.tolist()[0]

            errors, diff = count_errors(midi_seq, predicted_seq)

            source_notes = [note for note, _ in midi_seq]
            source_times = [time for _, time in midi_seq]
            predicted_notes = [note for note, _ in predicted_seq]
            predicted_times = [time for _, time in predicted_seq]

            all_source_notes.extend(source_notes)
            all_predicted_notes.extend(predicted_notes)
            all_source_times.extend(source_times)
            all_predicted_times.extend(predicted_times)

            print("Predicted: ", predicted_seq)
            print("Source:    ", midi_seq)
            print(f"Errors: {errors}/{max_seq_len}")
            print(f"Difference: {diff}")
            print()

            predicted_midi_seq = time_series_to_midi_seq(predicted_seq, mode="new")
            source_midi_seq = time_series_to_midi_seq(midi_seq, mode="new")

            df_results = calculate_measures(predicted_midi_seq, source_midi_seq)
            df_final_results = pd.concat([df_final_results, df_results], ignore_index=True)

    df_final_results.to_csv(csv_file_name, index=True)

    cm_notes = confusion_matrix(all_source_notes, all_predicted_notes)
    cm_times = confusion_matrix(all_source_times, all_predicted_times)

    def plot_confusion_matrix(cm, type):
        plt.figure(figsize=[6, 5])
        sns.heatmap(cm, annot=True, fmt='d', linewidths=0.5, norm=mcolors.AsinhNorm(), cmap="BuPu", cbar=False)
        plt.yticks(rotation=0, va='center')
        plt.xlabel(f"Predicted {type}")
        plt.ylabel(f"True {type}")
        plt.title(f"Confusion Matrix for model v.{version_name}")
        plt.show()

    plot_confusion_matrix(cm_notes, "Notes")
    plot_confusion_matrix(cm_times, "Times")


if __name__ == '__main__':
    validate()