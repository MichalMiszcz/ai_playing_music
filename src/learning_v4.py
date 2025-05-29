import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import mido

from src.cnnrnn_model import CNNRNNModel
from src.music_image_dataset import MusicImageDataset

image_root = "data/images"
midi_root = "data/processed_midi"
selected_image_path = "data/images/albeniz/alb_esp1/alb_esp1-1.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_transform = transforms.Compose([
    transforms.Resize((992, 1402)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_model(model, dataloader, epochs=50, device=device, learning_rate=0.0005):
    learning_data = []

    model = model.to(device)
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (images, midi_batch) in enumerate(dataloader):
            images = images.to(device)
            midi_batch = midi_batch.to(device)

            optimizer.zero_grad()
            output = model(images, midi_batch)
            loss = criterion_mse(output, midi_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.6f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        learning_data.append((epoch, avg_loss))

    torch.save(model.state_dict(), 'model_mini.pth')
    print("Model saved as 'model_mini.pth'")
    return learning_data

def generate_midi_from_selected_image(model, dataset, image_path, output_midi_path, device=device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    transform = image_transform
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        predicted_sequence = output[0].cpu().detach().numpy().tolist()
        print(f"Raw predicted sequence: {predicted_sequence[:10]}")
        predicted_sequence = [(int(p * 127 + 0.5), int(d * dataset.max_duration + 0.5)) for p, d in predicted_sequence]
        print(f"Scaled predicted sequence: {predicted_sequence[:10]}")
        sequence_to_midi(predicted_sequence, output_midi_path)

def sequence_to_midi(sequence, output_midi_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for pitch, duration in sequence:
        if pitch > 0:
            pitch = max(0, min(127, pitch))
            duration = max(0, duration)
            track.append(mido.Message('note_on', note=pitch, velocity=64, time=0))
            track.append(mido.Message('note_off', note=pitch, velocity=0, time=duration))

    mid.save(output_midi_path)


def generate_chart(data):
    epochs = [t[0] for t in data]
    avg_losses = [t[1] for t in data]

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, avg_losses, marker='o', linestyle='-', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    max_seq_len = 10
    dataset = MusicImageDataset(image_root, midi_root, image_transform, max_seq_len=max_seq_len, max_midi_files=30) # więcej plików
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    model = CNNRNNModel(input_channels=3, hidden_dim=512, output_dim=2, max_seq_len=max_seq_len)
    learning_data = train_model(model, dataloader, epochs=25, device=device, learning_rate=0.00002) # więcej epok, mniejszy lr

    generate_chart(learning_data)

    generate_midi_from_selected_image(model, dataset, selected_image_path, "output_selected_from_main.mid")