import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import mido

from global_variables import *

from src.music_program.cnnrnn_model_4_greyscale import CNNRNNModel
from src.music_program.music_image_dataset_4_greyscale import MusicImageDataset

image_root = "src/all_data/generated/my_complex_images/my_midi_images"
midi_root = "src/all_data/generated/generated_complex_midi_processed"
image_root_test = "src/all_data/generated/my_complex_images_test/my_midi_images"
midi_root_test = "src/all_data/generated/generated_complex_midi_processed_test"
selected_image_path = "src/all_data/generated/my_complex_images/my_midi_images/my_midi_files/song_1/song_1-1.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.RandomAffine(degrees=0, shear=2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.5], std=[0.5])
])

def train_model(model, dataloader, val_dataloader, epochs=50, device=device, learning_rate=0.0005, weight_decay=0.00001, max_norm=1.0, milestones=[100, 200, 300], lr_patience=6, es_patience=14):
    learning_data = []
    learning_data_val = []

    model = model.to(device)
    criterion = torch.nn.HuberLoss(delta=1.0)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=lr_patience)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                steps_per_epoch=len(dataloader), epochs=epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, (images, midi_batch) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            midi_batch = midi_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(images, midi_batch)
            loss = criterion(output, midi_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, midi_batch in val_dataloader:
                images, midi_batch = images.to(device, non_blocking=True), midi_batch.to(device, non_blocking=True)
                outputs = model(images)
                val_loss += criterion(outputs, midi_batch).item()
        val_loss /= len(val_dataloader)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.6f}")
        learning_data.append((epoch, avg_loss))
        learning_data_val.append((epoch, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), '/content/drive/MyDrive/modele_ai/model_best.pth')
            print("Model saved as 'model_best.pth'")
        else:
            patience_counter += 1

        if patience_counter >= es_patience:
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
    max_seq_len = 96
    left_hand_tracks = ['Piano left', 'Left']
    right_hand_tracks = ['Piano right', 'Right', 'Track 0']
    dataset = MusicImageDataset(image_root, midi_root, left_hand_tracks, right_hand_tracks, image_transform,
                                max_seq_len=max_seq_len, max_midi_files=10240, modify_image=True)
    val_dataset = MusicImageDataset(image_root_test, midi_root_test, left_hand_tracks, right_hand_tracks, image_transform, max_seq_len=max_seq_len, max_midi_files=1024, modify_image=True)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=8, pin_memory=True)

    model = CNNRNNModel(input_channels=1, hidden_dim=256, outp


ut_dim=3, max_seq_len=max_seq_len, rnn_layers=4)
    epochs = 300
    learning_data, learning_data_val = train_model(model, dataloader, val_dataloader, epochs=epochs, device=device, learning_rate=0.0001, weight_decay=0.0001, max_norm=1.0)

    generate_chart(learning_data, 'Training Loss over Epochs')
    generate_chart(learning_data_val, 'Validation Loss over Epochs')