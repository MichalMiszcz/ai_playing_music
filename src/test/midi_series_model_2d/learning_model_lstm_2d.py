"""
Implementacja uczenia modelu przy jednowymiarowym wyjściu.

Wyjściowa sekwencja wygląda w następujący sposób:
[64, 64, 65, 64, 64, 64, 64, 62, 62, 62, 62, 62, 62, ... , 60]
"""
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.music_program.utils.global_variables import *

from src.test.midi_series_model_2d.model import ModelLSTM
from src.test.midi_series_model_2d.dataset import MusicSequenceDataset

max_seq_len = 96
max_series_len = int(max_seq_len / 2)

max_midi_files=4096
max_midi_files_test=512
batch_size=32
hidden_dim=128
rnn_layers=2

epochs=100
learning_rate = 0.001
weight_decay = 0.00001
max_norm=1.0

image_root = "src/all_data/generated/my_complex_images/my_midi_images"
midi_root = "src/all_data/generated/generated_complex_midi_processed"
image_root_test = "src/all_data/generated/my_complex_images_test/my_midi_images"
midi_root_test = "src/all_data/generated/generated_complex_midi_processed_test"
selected_image_path = "src/all_data/generated/my_complex_images/my_midi_images/my_midi_files/song_1/song_1-1.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    # transforms.RandomAffine(degrees=0, shear=2),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.5], std=[0.5])
])

class LpLoss(nn.Module):
    def __init__(self, p=3.0):
        super(LpLoss, self).__init__()
        self.p = p

    def forward(self, y_pred, y_true):
        loss = torch.mean(torch.abs(y_pred - y_true) ** self.p)
        return loss

def train_model(model, dataloader, val_dataloader, epochs=50, device=device, learning_rate=0.0005, weight_decay=0.00001, max_norm=1.0, milestones=[100, 200, 300], lr_patience=6, es_patience=14, teacher_epochs=2):
    learning_data = []
    learning_data_val = []

    model = model.to(device)
    criterion = torch.nn.MSELoss() #LpLoss(1.5) #torch.nn.MSELoss() #torch.nn.HuberLoss(delta=1.0)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=lr_patience)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3,
                                                # steps_per_epoch=len(dataloader), epochs=epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_val_loss = float("inf")
    patience = es_patience
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # epochs_ratio = epoch/epochs
        if epoch > teacher_epochs:
            teacher_ratio = max(0.0, 0.75 - (epoch / epochs))
        else:
            teacher_ratio = 1.0

        for i, (midi_batch, midi_series) in enumerate(dataloader):
            midi_batch, midi_series = midi_batch.to(device, non_blocking=True), midi_series.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(midi_batch, midi_series, teacher_ratio)

            loss = criterion(output, midi_series)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for midi_batch, midi_series in val_dataloader:
                midi_batch, midi_series = midi_batch.to(device, non_blocking=True), midi_series.to(device, non_blocking=True)
                outputs = model(midi_batch)

                val_loss += criterion(outputs, midi_series).item()
        val_loss /= len(val_dataloader)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.6f}")
        learning_data.append((epoch, avg_loss))
        learning_data_val.append((epoch, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'src/model_lstm_best.pth')
            print("Model saved as 'model_lstm_best.pth'")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return learning_data, learning_data_val

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
    plt.show()


if __name__ == "__main__":
    left_hand_tracks = ['Piano left', 'Left']
    right_hand_tracks = ['Piano right', 'Right', 'Track 0']
    dataset = MusicSequenceDataset(image_root, midi_root, left_hand_tracks, right_hand_tracks, image_transform, max_seq_len=max_seq_len, max_series_len=max_series_len, max_midi_files=max_midi_files, modify_image=False)
    val_dataset = MusicSequenceDataset(image_root_test, midi_root_test, left_hand_tracks, right_hand_tracks, image_transform, max_seq_len=max_seq_len, max_series_len=max_series_len, max_midi_files=max_midi_files_test, modify_image=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, pin_memory=True)

    model = ModelLSTM(input_dim=3, hidden_dim=hidden_dim, output_dim=2, max_seq_len=max_seq_len, max_series_len=max_series_len, rnn_layers=rnn_layers)
    model = torch.compile(model)
    epochs = epochs
    learning_data, learning_data_val = train_model(model, dataloader, val_dataloader, epochs=epochs, device=device, learning_rate=learning_rate, weight_decay=weight_decay, lr_patience=3, es_patience=13, teacher_epochs=5)

    generate_chart(learning_data, 'Training Loss over Epochs')
    generate_chart(learning_data_val, 'Validation Loss over Epochs')