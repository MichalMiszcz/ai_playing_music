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

image_root = "src/all_data/generated/my_images/my_midi_images"
midi_root = "src/all_data/generated/generated_songs_processed"
# image_root_test = "my_images_test/my_midi_images"
# midi_root_test = "generated_songs_processed_test"
selected_image_path = "src/all_data/generated/my_images/my_midi_images/my_midi_files/song_1/song_1-1.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_transform = transforms.Compose([
    transforms.Resize((SIZE_X, SIZE_Y)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def train_model(model, dataloader, epochs=50, device=device, learning_rate=0.0005, weight_decay=0.00001, max_norm=1.0):
    learning_data = []

    model = model.to(device)
    criterion = torch.nn.HuberLoss(delta=1.0)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 290], gamma=0.1)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (images, midi_batch) in enumerate(dataloader):
            images = images.to(device)
            midi_batch = midi_batch.to(device)

            optimizer.zero_grad()
            output = model(images, midi_batch)
            loss = criterion(output, midi_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            optimizer.step()
            total_loss += loss.item()

            print("loss:", loss.item())
            if (i + 1) % 128 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        learning_data.append((epoch, avg_loss))

    # torch.save(model.state_dict(), '/content/drive/MyDrive/modele_ai/model_bitmap.pth')
    torch.save(model.state_dict(), 'model_mini.pth')
    print("Model saved as 'model_mini.pth'")
    return learning_data

def test_model(model, val_dataloader, device=device):
    criterion = torch.nn.HuberLoss(delta=1.0)

    total_val_loss = 0

    for i, (images, midi_batch) in enumerate(val_dataloader):
        images = images.to(device)
        midi_batch = midi_batch.to(device)
        output = model(images, midi_batch)
        val_loss = criterion(output, midi_batch)

        total_val_loss += val_loss

    avg_val_loss = total_val_loss / len(val_dataloader)

    print(f"Validation Loss: {avg_val_loss:.6f}")

def generate_midi_from_selected_image(model, dataset, image_path, output_midi_path, device=device):
    model.eval()
    image = Image.open(image_path).convert('L')
    transform = image_transform
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        predicted_sequence = output[0].cpu().detach().numpy().tolist()
        print(f"Raw predicted sequence: {predicted_sequence[:32]}")

        final_predicted_sequence = []
        for norm_note, norm_vel, norm_dt in predicted_sequence:
            note_idx = int(norm_note * (NUM_NOTES - 1.0) + 0.5)
            note_idx = max(0, min(note_idx, NUM_NOTES - 1))
            midi_note = WHITE_KEYS_MIDI[note_idx]

            velocity_idx = int(norm_vel * (NUM_VELOCITIES - 1.0) + 0.5)
            velocity_idx = max(0, min(velocity_idx, NUM_VELOCITIES - 1))
            velocity = VELOCITY[velocity_idx]

            delta_time_idx = int(norm_dt * (NUM_DELTA_TIME - 1.0) + 0.5)
            delta_time_idx = max(0, min(delta_time_idx, NUM_DELTA_TIME - 1.0))
            delta_time = DELTA_TIME[delta_time_idx]

            final_predicted_sequence.append((midi_note, velocity, delta_time))

        final_predicted_sequence = final_predicted_sequence[:32]

        print(f"Scaled predicted sequence: {final_predicted_sequence}")
        sequence_to_midi(final_predicted_sequence, output_midi_path)

def sequence_to_midi(sequence, output_midi_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    right_track = mido.MidiTrack()

    current_time = 0
    events = []

    for note, velocity, delta_time in sequence:
        current_time += delta_time
        events.append((current_time, note, velocity))

    right_events = events

    right_events.sort(key=lambda x: x[0])

    def add_events_to_track(track, events):
        prev_time = 0
        for time, note, velocity in events:
            delta_time = time - prev_time
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta_time))
            prev_time = time

    add_events_to_track(right_track, right_events)

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
    max_seq_len = 32
    left_hand_tracks = ['Piano left', 'Left']
    right_hand_tracks = ['Piano right', 'Right', 'Track 0']
    dataset = MusicImageDataset(image_root, midi_root, left_hand_tracks, right_hand_tracks, image_transform,
                                max_seq_len=max_seq_len, max_midi_files=128)
    # val_dataset = MusicImageDataset(image_root_test, midi_root_test, left_hand_tracks, right_hand_tracks, image_transform, max_seq_len=max_seq_len, max_midi_files=4)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # val_dataloader = DataLoader(dataset, shuffle=True)

    model = CNNRNNModel(input_channels=1, hidden_dim=64, output_dim=3, max_seq_len=max_seq_len, rnn_layers=1)
    learning_data = train_model(model, dataloader, epochs=5, device=device, learning_rate=0.1, weight_decay=0.0001, max_norm=1.0)

    generate_chart(learning_data)

    generate_midi_from_selected_image(model, dataset, selected_image_path, "output_selected_from_main.mid")