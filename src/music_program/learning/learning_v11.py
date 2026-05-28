"""
Uczenie modelu bazującego na sieci konwolucyjnej. Dane uczące to obrazy zawierające tylko jedną pięciolinię oraz dwuwymiarowe MIDI.
"""
# Biblioteki
import pandas as pd
import time
from PIL import Image
from torchvision.transforms.v2.functional import to_pil_image

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.music_program.learning import params_dict
from src.music_program.utils.global_variables import *

from src.music_program.model.cnn_model_v11 import MusicModel
from src.music_program.dataset.music_image_dataset_11 import MusicImageDataset
from src.utils.python_colors import bcolors
from src.utils.teacher_ratio import count_teacher_ratio

# Parametry modelu i uczenia
model_dir = None #'src/_models/image_to_midi/model_best_v800_001_checkpoint.pth'

version = 900
# subversion = None
#
max_seq_len = 64
max_series_len = int(max_seq_len / 2)
#
# max_midi_files = 4096
# max_midi_files_test = 1024
# batch_size = 16
val_batch_size = 512
# features_number = 8

epochs = 100
# learning_rate = 0.0001
weight_decay = 0.00001
max_norm = 1.0

lr_patience = 6
es_patience = 25

# version_name = str(version) + '_' + str(subversion) if subversion is not None else str(version)
# print(f'Version name: {version_name}')

image_root = "src/all_data/generated/my_complex_images/my_midi_images"
midi_root = "src/all_data/generated/generated_complex_midi_processed"
image_root_test = "src/all_data/generated/my_complex_images_test/my_midi_images"
midi_root_test = "src/all_data/generated/generated_complex_midi_processed_test"
selected_image_path = "src/all_data/generated/my_complex_images/my_midi_images/my_midi_files/song_1/song_1-1.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_transform = v2.Compose([
    v2.Resize((HEIGHT, WIDTH)),
    # v2.RandomAffine(degrees=1, shear=0),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomInvert(p=0.5),
    v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5)
])

image_validation_transform = v2.Compose([
    v2.Resize((HEIGHT, WIDTH)),
    # v2.RandomAffine(degrees=1, shear=0),
    # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # v2.RandomInvert(p=0.5),
    # v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5)
])


def checkpoint(model_to_save, optimizer, scheduler, epoch, val_loss, filename):
    torch.save(model_to_save.state_dict(), filename)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            }, filename)

def resume(model_to_load_to, optimizer, scheduler, filename):
    checkpoint = torch.load(filename)
    model_to_load_to.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']

    print(f"Previous best epoch: {epoch}")
    print(f"Previous best validation loss: {loss}")

    return epoch, loss


# Uczenie modelu
def train_model(model, dataloader, val_dataloader, epochs=50, device=device, learning_rate=0.0005, weight_decay=0.00001,
                max_norm=1.0, milestones=[100, 200, 300], lr_patience=6, es_patience=14,
                mixed_teacher_forcing_epochs: list[int] | None = None):
    learning_data = []
    learning_data_val = []

    additional_learning_data = {}

    model = model.to(device)
    # criterion = torch.nn.HuberLoss(delta=1.0)  # LpLoss(1.5) torch.nn.MSELoss() #torch.nn.HuberLoss(delta=1.0)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=lr_patience)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(dataloader), epochs=epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 200, 350, 500], gamma=0.3)

    # Loading existing model
    if model_dir:
        print("Loading model: ", model_dir)
        resume(model, optimizer, scheduler, model_dir)
    else:
        print("Learning new model")

    best_val_loss = float("inf")
    patience = es_patience
    patience_counter = 0
    last_ep = 0

    start = time.asctime()
    print()
    print(f'Start time: {start}')
    print('─' * 100)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, (images, midi_batch) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            midi_batch = midi_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            output_notes, output_times = model(images)

            # print("Output shape:", output.shape)
            # print("Output:", output)
            # print("Midi shape:", midi_batch.shape)
            # print("Midi:", midi_batch)
            # print(output)
            # print(midi_batch)

            # print(output.shape)
            # print(midi_batch.shape)

            loss_note = criterion(output_notes.reshape(-1, 9), midi_batch[:, :, 0].reshape(-1))
            loss_time = criterion(output_times.reshape(-1, 6), midi_batch[:, :, 1].reshape(-1))
            loss = loss_note + loss_time

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            optimizer.step()
            # scheduler.step() # OneCycleLR

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, midi_batch in val_dataloader:
                images = images.to(device, non_blocking=True)
                midi_batch = midi_batch.to(device, non_blocking=True)

                output_notes, output_times = model(images)

                loss_note = criterion(output_notes.reshape(-1, 9), midi_batch[:, :, 0].reshape(-1))
                loss_time = criterion(output_times.reshape(-1, 6), midi_batch[:, :, 1].reshape(-1))
                loss = loss_note + loss_time

                val_loss += loss.item()
        val_loss /= len(val_dataloader)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}")
        print(f"Epoch {epoch + 1}/{epochs}, {bcolors.OKBLUE}Validation Loss: {val_loss:.6f}{bcolors.ENDC}")
        learning_data.append((epoch, avg_loss))
        learning_data_val.append((epoch, val_loss))

        if (epoch + 1) % 25 == 0:
            additional_learning_data[f'val_loss_e-{epoch + 1}'] = val_loss
            additional_learning_data[f'train_loss_e-{epoch + 1}'] = avg_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            additional_learning_data['best_val_loss'] = best_val_loss
            additional_learning_data['best_epoch'] = epoch + 1

            patience_counter = 0
            # torch.save(model.state_dict(), f'src/_models/image_to_midi/model_best_v{version_name}.pth')
            checkpoint(model, optimizer, scheduler, epoch + 1, val_loss, f'src/_models/image_to_midi/model_best_v{version_name}_checkpoint.pth')
            print(f"Model saved as {bcolors.OKGREEN}'src/_models/image_to_midi/model_best_v{version_name}.pth'{bcolors.ENDC}")
        else:
            patience_counter += 1

        last_ep = epoch + 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            additional_learning_data['last_val_loss'] = val_loss
            additional_learning_data['last_loss'] = avg_loss
            break

        if (epoch + 1) % 10 == 0:
            generate_chart(learning_data, f'Training Loss over Epochs', epoch + 1)
            generate_chart(learning_data_val, f'Validation Loss over Epochs', epoch + 1)

    end = time.asctime()
    print('─' * 100)
    print(f'End time: {end}')

    additional_learning_data['start_time'] = start
    additional_learning_data['end_time'] = end

    additional_learning_data['last_epoch'] = last_ep
    return learning_data, learning_data_val, additional_learning_data


# Generowanie wykresu
def generate_chart(data, title, epoch = None, mode = "save"):
    name = title
    if epoch is not None:
        title = title + f" - epoch {epoch}"

    epochs = [t[0] for t in data]
    avg_losses = [t[1] for t in data]

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    plt.plot(epochs, avg_losses, marker='o', linestyle='-', color='blue')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'src/plots/{name}_v{version_name}.png')
    plt.close()

    if mode == "show":
        img = Image.open(f'src/plots/{title}_v{version_name}.png')
        img.show()

# Uruchomienie uczenia
if __name__ == "__main__":
    params_dictionary = params_dict
    for params in params_dictionary.params_dict:
        subversion = params['subversion']
        max_midi_files = params['max_midi_files']
        max_midi_files_test = params['max_midi_files_test']
        batch_size = params['batch_size']
        features_number = params['features_number']
        hidden_dim = params['hidden_dim']
        learning_rate = params['learning_rate']

        version_name = str(version) + '_' + str(subversion) if subversion is not None else str(version)
        print(f'Version name: {version_name}')

        left_hand_tracks = ['Piano left', 'Left']
        right_hand_tracks = ['Piano right', 'Right', 'Track 0']

        dataset = MusicImageDataset(image_root, midi_root, left_hand_tracks, right_hand_tracks, image_transform,
                                    max_seq_len=max_seq_len, max_series_len=max_series_len, max_midi_files=max_midi_files,
                                    modify_image=False)
        val_dataset = MusicImageDataset(image_root_test, midi_root_test, left_hand_tracks, right_hand_tracks,
                                        image_validation_transform, max_seq_len=max_seq_len, max_series_len=max_series_len,
                                        max_midi_files=max_midi_files_test, modify_image=False)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True)
        # val_dataloader = dataloader

        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # val_dataloader = DataLoader(val_dataset, shuffle=False)

        model = MusicModel(features_number, hidden_dim, max_series_len)

        epochs = epochs
        learning_data, learning_data_val, additional_learning_data = train_model(model, dataloader, val_dataloader, epochs=epochs, device=device,
                                                       learning_rate=learning_rate, weight_decay=weight_decay,
                                                       lr_patience=lr_patience, es_patience=es_patience)

        additional_learning_data_pd = pd.DataFrame.from_dict(additional_learning_data, orient="index", columns=["Value"])
        additional_learning_data_pd.to_csv(f'src/csv/additional_learning_data_v{version_name}.csv')

        generate_chart(learning_data, 'Training Loss over Epochs')
        generate_chart(learning_data_val, 'Validation Loss over Epochs')
