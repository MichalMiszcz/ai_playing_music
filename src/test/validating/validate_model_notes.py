import os

import mido
import torch
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.music_program.learning import params_dict
from src.music_program.utils.global_variables import *

from src.music_program.model.cnn_model_v10 import MusicModel
from src.music_program.dataset.music_image_dataset_10 import MusicImageDataset

note_to_index = {midi_num: i for i, midi_num in enumerate(WHITE_KEYS_MIDI)}
velocity_to_index = {midi_num: i for i, midi_num in enumerate(VELOCITY)}
delta_time_to_index = {midi_num: i for i, midi_num in enumerate(DELTA_TIME)}

model_path = "src/_models/image_to_midi/model_best_v800_5.pth"
image_root_test = "src/all_data/generated/my_complex_images_test/my_midi_images"
midi_root_test = "src/all_data/generated/generated_complex_midi_processed_test"

midi_columns = ['midi_note', 'velocity', 'delta_time']



version = 800
subversion = 9

max_seq_len = 64
max_series_len = 16

batch_size = 32
features_number = 32
hidden_dim = 256


version_name = str(version) + '_' + str(subversion) if subversion is not None else str(version)
print(f'Version name: {version_name}')

max_midi_files_test = 32

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


def main():

    model_mode = "my_model"
    res = "low"
    max_seq_len = 96

    model = MusicModel(features_number, hidden_dim, max_series_len)

    model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    for i, (images, midi_batch) in enumerate(val_dataloader):
        with torch.no_grad():
            images = images.to(device)
            midi_batch = midi_batch.to(device)

            output = model(images)
            predicted_sequence = output[0].cpu().detach().numpy().tolist()
            predicted_sequence = predicted_sequence[:max_seq_len]
            notes = [note_logit.index(max(note_logit)) for note_logit in predicted_sequence]

            midi_batch = midi_batch.cpu()
            midi_batch = midi_batch.tolist()
            midi_batch = midi_batch[0]

            midi_notes = [round(note * 1) for note in midi_batch]
            predicted_notes = [round(note * 1) for note in notes]

            print("Predicted: ", predicted_notes)
            print("Source:    ", midi_notes)
            print()


if __name__ == '__main__':
    main()
