import os
from sklearn.metrics import confusion_matrix

import mido
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import to_pil_image

from src.music_program.utils.global_variables import *

from src.music_program.model.cnn_model_v11_val import MusicModel
from src.music_program.dataset.music_image_dataset_11 import MusicImageDataset

note_to_index = {midi_num: i for i, midi_num in enumerate(WHITE_KEYS_MIDI)}
velocity_to_index = {midi_num: i for i, midi_num in enumerate(VELOCITY)}
delta_time_to_index = {midi_num: i for i, midi_num in enumerate(DELTA_TIME)}

model_path = "src/_models/image_to_midi/model_best_v900_002_checkpoint.pth"
image_root_test = "src/all_data/generated/my_complex_images_test/my_midi_images"
midi_root_test = "src/all_data/generated/generated_complex_midi_processed_test"

midi_columns = ['midi_note', 'velocity', 'delta_time']


version = 900
subversion = '002'

max_seq_len = 64
max_series_len = 32

batch_size = 32
features_number = 32
hidden_dim = 64


version_name = str(version) + '_' + str(subversion) if subversion is not None else str(version)
print(f'Version name: {version_name}')

max_midi_files_test = 78

left_hand_tracks = ['Piano left', 'Left']
right_hand_tracks = ['Piano right', 'Right', 'Track 0']

image_transform = v2.Compose([
    v2.Resize((HEIGHT, WIDTH)),
    # v2.RandomAffine(degrees=1, shear=0),
    # v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # v2.RandomInvert(p=1.0),
    # v2.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0)
])

# Dataset
val_dataset = MusicImageDataset(image_root_test, midi_root_test, left_hand_tracks, right_hand_tracks,
                                image_transform, max_seq_len=max_seq_len, max_series_len=max_series_len,
                                max_midi_files=max_midi_files_test, modify_image=False)
val_dataloader = DataLoader(val_dataset, shuffle=False, pin_memory=True)

# Loading model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_errors(source, output):
    number_of_errors = 0
    sum_difference = 0

    for i, [note, time] in enumerate(source):
        if output[i][0] != note:
            number_of_errors += 1
            sum_difference += abs(note - output[i][0])

        if output[i][1] != time:
            number_of_errors += 1
            sum_difference += abs(time - output[i][1])

    return number_of_errors, sum_difference

def main():
    model = MusicModel(features_number, hidden_dim, max_series_len)

    model.to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_source_notes = []
    all_predicted_notes = []
    all_source_times = []
    all_predicted_times = []
    for i, (images, midi_batch) in enumerate(val_dataloader):
        with torch.no_grad():
            images = images.to(device)
            midi_batch = midi_batch.to(device)

            output_notes, output_times = model(images)
            predicted_notes = output_notes[0].cpu().detach().numpy().tolist()
            predicted_times = output_times[0].cpu().detach().numpy().tolist()
            predicted_notes = predicted_notes[:max_seq_len]
            predicted_times = predicted_times[:max_seq_len]
            notes = [note_logit.index(max(note_logit)) for note_logit in predicted_notes]
            times = [note_logit.index(max(note_logit)) for note_logit in predicted_times]

            midi_batch = midi_batch.cpu()
            midi_batch = midi_batch.tolist()
            midi = midi_batch[0]
            predicted = list(map(list, zip(notes, times)))

            errors, diff = count_errors(midi, predicted)

            source_notes = [note for note, _ in midi]
            source_times = [time for _, time in midi]
            all_source_notes.extend(source_notes)
            all_predicted_notes.extend(notes)
            all_source_times.extend(source_times)
            all_predicted_times.extend(times)

            print("Predicted: ", predicted)
            print("Source:    ", midi)
            print(f"Errors: {errors}/{max_series_len}")
            print(f"Difference: {diff}")
            print()

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
    main()
