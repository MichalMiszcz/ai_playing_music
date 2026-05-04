"""
Implementacja uczenia modelu przy jednowymiarowym wyjściu.

Wyjściowa sekwencja wygląda w następujący sposób:
[(64, 20160), (72, 20160), (72, 5040), (65, 10080), (64, 5040), ...]
"""

# Biblioteki
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.music_program.utils.global_variables import *

from src.music_program.model.cnnrnn_model_7_2 import CNNRNNModel
from src.music_program.dataset.music_image_dataset_7_2 import MusicImageDataset
from src.utils.teacher_ratio import count_teacher_ratio
from src.utils.python_colors import bcolors

# Parametry modelu i uczenia
version = 3003
subversion = None

max_seq_len = 96
max_series_len = int(max_seq_len / 2)

max_midi_files = 2048
max_midi_files_test = 256
batch_size = 4
hidden_dim = 64
emb_dim_note = 8
emb_dim_delta_time = 8
rnn_layers = 2

epochs = 100
learning_rate = 0.01 #0.001
weight_decay = 0.0001
max_norm = 1.0

lr_patience = 5
es_patience = 15
mixed_teacher_forcing_epochs = [5, 50]

# lr_patience = 15
# es_patience = 25
# mixed_teacher_forcing_epochs = [0, 0]

model_dir = None #'src/_models/image_to_midi/model_best_v3002_1.pth'

version_name = str(version) + '_' + str(subversion) if subversion is not None else str(version)
print(f'Version name: {version_name}')

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
    # v2.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0)
])


# Własna funkcja straty
class LpLoss(nn.Module):
    def __init__(self, p=3.0):
        super(LpLoss, self).__init__()
        self.p = p

    def forward(self, y_pred, y_true):
        loss = torch.mean(torch.abs(y_pred - y_true) ** self.p)
        return loss


# Uczenie modelu
def train_model(model, dataloader, val_dataloader, epochs=50, device=device, learning_rate=0.0005, weight_decay=0.00001,
                max_norm=1.0, milestones=[100, 200, 300], lr_patience=6, es_patience=14,
                mixed_teacher_forcing_epochs: list[int] | None = None):
    learning_data = []
    learning_data_val = []

    additional_learning_data = {}

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()  # LpLoss(1.5) torch.nn.MSELoss() #torch.nn.HuberLoss(delta=1.0)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=lr_patience)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3,
    # steps_per_epoch=len(dataloader), epochs=epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 200, 350, 500], gamma=0.3)

    best_val_loss = float("inf")
    patience = es_patience
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_note_loss = 0
        total_time_loss = 0

        # epochs_ratio = epoch/epochs
        first_epoch = mixed_teacher_forcing_epochs[0]
        last_epoch = mixed_teacher_forcing_epochs[1]
        teacher_ratio = count_teacher_ratio(epoch, first_epoch, last_epoch)

        for i, (images, midi_batch) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            midi_batch = midi_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            # output = model(images, midi_batch, teacher_ratio)
            output_notes, output_delta_time = model(images, midi_batch, teacher_ratio)

            loss_note = criterion(output_notes.transpose(1, 2), midi_batch[:, :, 0].long())
            loss_delta_time = criterion(output_delta_time.transpose(1, 2), midi_batch[:, :, 1].long())

            sum_loss = loss_note + loss_delta_time
            sum_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            optimizer.step()
            total_loss += sum_loss.item()
            total_note_loss += loss_note.item()
            total_time_loss += loss_delta_time.item()

            # print("loss:", loss.item())
            # if (i + 1) % 128 == 0:
            #     print(f"Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        avg_note_loss = total_note_loss / len(dataloader)
        avg_time_loss = total_time_loss / len(dataloader)

        model.eval()
        val_loss = 0
        total_note_loss = 0
        total_time_loss = 0
        with torch.no_grad():
            for images, midi_batch in val_dataloader:
                images, midi_batch = images.to(device, non_blocking=True), midi_batch.to(device, non_blocking=True)
                output_notes, output_delta_time = model(images)

                loss_note = criterion(output_notes.transpose(1, 2), midi_batch[:, :, 0].long())
                loss_delta_time = criterion(output_delta_time.transpose(1, 2), midi_batch[:, :, 1].long())

                sum_loss = loss_note + loss_delta_time

                val_loss += sum_loss.item()
                total_note_loss += loss_note.item()
                total_time_loss += loss_delta_time.item()
        val_loss /= len(val_dataloader)
        val_note_loss = total_note_loss / len(dataloader)
        val_time_loss = total_time_loss / len(dataloader)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}, Average Note Loss: {avg_note_loss:.6f}, Average Time Loss: {avg_time_loss:.6f}")
        print(f"Epoch {epoch + 1}/{epochs}, {bcolors.OKBLUE}Validation Loss: {val_loss:.6f}{bcolors.ENDC}, Average Note Loss: {val_note_loss:.6f}, Average Time Loss: {val_time_loss:.6f}")
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

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            additional_learning_data['last_val_loss'] = val_loss
            additional_learning_data['last_epoch'] = epoch + 1
            break

    additional_learning_data['last_epoch'] = epochs
    return learning_data, learning_data_val


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
    left_hand_tracks = ['Piano left', 'Left']
    right_hand_tracks = ['Piano right', 'Right', 'Track 0']

    dataset = MusicImageDataset(image_root_test, midi_root_test, left_hand_tracks, right_hand_tracks, image_transform,
                                max_seq_len=max_seq_len, max_series_len=max_series_len, max_midi_files=max_midi_files,
                                modify_image=False)

    # dataset = MusicImageDataset(image_root, midi_root, left_hand_tracks, right_hand_tracks, image_transform,
    #                             max_seq_len=max_seq_len, max_series_len=max_series_len, max_midi_files=max_midi_files,
    #                             modify_image=False)
    # val_dataset = MusicImageDataset(image_root_test, midi_root_test, left_hand_tracks, right_hand_tracks,
    #                                 image_transform, max_seq_len=max_seq_len, max_series_len=max_series_len,
    #                                 max_midi_files=max_midi_files_test, modify_image=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader = dataloader

    # val_dataloader = DataLoader(val_dataset, shuffle=False, pin_memory=True)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, shuffle=False)

    model = CNNRNNModel(input_channels=1, hidden_dim=hidden_dim, emb_dim_note=emb_dim_note,
                        emb_dim_delta_time=emb_dim_delta_time, output_dim=2, max_seq_len=max_seq_len,
                        max_series_len=max_series_len, rnn_layers=rnn_layers)

    if model_dir:
        print("Loading model: ", model_dir)
        model.load_state_dict(torch.load(model_dir, map_location=device, weights_only=True))
    else:
        print("Learning new model")

    epochs = epochs
    learning_data, learning_data_val = train_model(model, dataloader, val_dataloader, epochs=epochs, device=device,
                                                   learning_rate=learning_rate, weight_decay=weight_decay,
                                                   lr_patience=lr_patience, es_patience=es_patience,
                                                   mixed_teacher_forcing_epochs=mixed_teacher_forcing_epochs)

    generate_chart(learning_data, 'Training Loss over Epochs')
    generate_chart(learning_data_val, 'Validation Loss over Epochs')
