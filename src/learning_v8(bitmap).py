import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import mido

from src.cnnrnn_model_3_bitmap import CNNRNNModel
from src.music_image_dataset_3_bitmap import MusicImageDataset
from src.global_variables import SIZE_X, SIZE_Y

image_root = "my_images/my_midi_images"
midi_root = "generated_songs_processed"
selected_image_path = "my_images/my_midi_images/song_1/song_1-1.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_transform = transforms.Compose([
    # transforms.Resize((SIZE_X, SIZE_Y)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def train_model(model, dataloader, epochs=50, device=device, learning_rate=0.0005, weight_decay=0.00001):
    learning_data = []

    model = model.to(device)
    criterion_mse = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            print("loss:", loss.item())
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
    image = Image.open(image_path).convert('1')
    transform = image_transform
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        predicted_sequence = output[0].cpu().detach().numpy().tolist()
        print(f"Raw predicted sequence: {predicted_sequence[:10]}")
        predicted_sequence = [
            (1 if h > 0.5 else 0, int(n * 127 + 0.5), int(v * 127 + 0.5), int(dt * dataset.max_delta_time + 0.5))
            for h, n, v, dt in predicted_sequence
        ]
        print(f"Scaled predicted sequence: {predicted_sequence[:10]}")
        sequence_to_midi(predicted_sequence, output_midi_path)

def sequence_to_midi(sequence, output_midi_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    left_track = mido.MidiTrack()
    right_track = mido.MidiTrack()

    current_time = 0
    events = []
    for hand, note, velocity, delta_time in sequence:
        current_time += delta_time
        events.append((current_time, hand, note, velocity))

    left_events = [e for e in events if e[1] == 0]
    right_events = [e for e in events if e[1] == 1]

    left_events.sort(key=lambda x: x[0])
    right_events.sort(key=lambda x: x[0])

    def add_events_to_track(track, events):
        prev_time = 0
        for time, _, note, velocity in events:
            delta_time = time - prev_time
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta_time))
            prev_time = time

    add_events_to_track(left_track, left_events)
    add_events_to_track(right_track, right_events)

    mid.tracks.append(left_track)
    mid.tracks.append(right_track)

    mid.save(output_midi_path)

def generate_chart(data):
    epochs = [t[0] for t in data]
    avg_losses = [t[1] for t in data]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, avg_losses, marker='o', linestyle='-', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    max_seq_len = 16
    left_hand_tracks = ['Piano left', 'Left']
    right_hand_tracks = ['Piano right', 'Right', 'Track 0']
    dataset = MusicImageDataset(image_root, midi_root, left_hand_tracks, right_hand_tracks, image_transform,
                                max_seq_len=max_seq_len, max_midi_files=512)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = CNNRNNModel(input_channels=1, hidden_dim=256, output_dim=3, max_seq_len=max_seq_len, rnn_layers=2)
    learning_data = train_model(model, dataloader, epochs=150, device=device, learning_rate=0.0005, weight_decay=0.0002)

    generate_chart(learning_data)

    generate_midi_from_selected_image(model, dataset, selected_image_path, "output_selected_from_main.mid")