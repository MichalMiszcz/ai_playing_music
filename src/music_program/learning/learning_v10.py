"""
Uczenie modelu bazującego na sieci konwolucyjnej. Dane uczące to obrazy zawierające tylko jedną pięciolinię oraz skrócone MIDI.
"""
# Biblioteki
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
from src.utils.python_colors import bcolors
from src.utils.teacher_ratio import count_teacher_ratio

# Parametry modelu i uczenia
version = 800
# subversion = None
#
max_seq_len = 64
max_series_len = 16 #int(max_seq_len / 2)
#
# max_midi_files = 4096
# max_midi_files_test = 1024
# batch_size = 16
# features_number = 8

epochs = 100
# learning_rate = 0.0001
weight_decay = 0.0001
max_norm = 1.0

lr_patience = 7
es_patience = 15

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
    # v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # v2.RandomInvert(p=1.0),
    # v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5)
])


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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=lr_patience)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3,
    # steps_per_epoch=len(dataloader), epochs=epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 200, 350, 500], gamma=0.3)

    best_val_loss = float("inf")
    patience = es_patience
    patience_counter = 0

    last_ep = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, (images, midi_batch) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            midi_batch = midi_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(images)

            # print("Output shape:", output.shape)
            # print("Output:", output)
            # print("Midi shape:", midi_batch.shape)
            # print("Midi:", midi_batch)
            # print(output)
            # print(midi_batch)

            # print(output.shape)
            # print(midi_batch.shape)

            loss = criterion(output.reshape(-1, 9), midi_batch.reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, midi_batch in val_dataloader:
                images = images.to(device, non_blocking=True)
                midi_batch = midi_batch.to(device, non_blocking=True)

                output = model(images)

                val_loss += criterion(output.reshape(-1, 9), midi_batch.reshape(-1)).item()
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
            torch.save(model.state_dict(), f'src/_models/image_to_midi/model_best_v{version_name}.pth')
            print(f"Model saved as {bcolors.OKGREEN}'src/model_best_v{version_name}.pth'{bcolors.ENDC}")
        else:
            patience_counter += 1

        last_ep = epoch + 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            additional_learning_data['last_val_loss'] = val_loss
            additional_learning_data['last_loss'] = avg_loss
            break

    additional_learning_data['last_epoch'] = last_ep
    return learning_data, learning_data_val, additional_learning_data


# Generowanie wykresu
def generate_chart(data, title):
    epochs = [t[0] for t in data]
    avg_losses = [t[1] for t in data]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, avg_losses, marker='o', linestyle='-', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'src/plots/{title}_v{version_name}.png')


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
                                        image_transform, max_seq_len=max_seq_len, max_series_len=max_series_len,
                                        max_midi_files=max_midi_files_test, modify_image=False)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
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
