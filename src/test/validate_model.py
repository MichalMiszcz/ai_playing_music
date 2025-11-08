import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.music_program.cnnrnn_model_4_greyscale import CNNRNNModel
from src.music_program.global_variables import *
from src.music_program.music_image_dataset_4_greyscale import MusicImageDataset
from src.test.accuracy import *

model_path = "model_new_bigeye.pth"
image_root_test = "all_data/generated/my_images_test/my_midi_images"
midi_root_test = "all_data/generated/generated_songs_processed_test"

max_seq_len = 32
max_midi_files = 16
left_hand_tracks = ['Piano left', 'Left']
right_hand_tracks = ['Piano right', 'Right', 'Track 0']


image_transform = transforms.Compose([
    # transforms.Resize((SIZE_X, SIZE_Y)), # Resizing image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset
val_dataset = MusicImageDataset(image_root_test, midi_root_test, left_hand_tracks, right_hand_tracks, image_transform, max_seq_len=max_seq_len, max_midi_files=max_midi_files)
val_dataloader = DataLoader(val_dataset, shuffle=True)

# Loading model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNRNNModel(input_channels=1, hidden_dim=512, output_dim=3, rnn_layers=3)
model.to(device)

model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()


def from_raw_to_midi(sequence):
    final_predicted_sequence = []
    for norm_note, norm_vel, norm_dt in sequence:
        note_idx = int(norm_note * (NUM_NOTES - 1.0) + 0.5)
        note_idx = max(0, min(note_idx, NUM_NOTES - 1))
        midi_note = WHITE_KEYS_MIDI[note_idx]

        velocity = int(norm_vel * 127.0 + 0.5)
        delta_time = int(norm_dt * 1008 + 0.5)

        final_predicted_sequence.append((midi_note, velocity, delta_time))

    final_predicted_sequence = final_predicted_sequence[:32]
    return final_predicted_sequence

def validate_predicted_midi(predicted: list, source: list, count):
    notes_predicted = [v for v, _, _ in predicted]
    notes_source = [v for v, _, _ in source]

    print("Mean absolute error: ", mean_absolute_error(notes_predicted, notes_source, max_seq_len))
    print("Mean square error: ", mean_square_error(notes_predicted, notes_source, max_seq_len))
    print("Root mean square error: ", root_mean_square_error(notes_predicted, notes_source, max_seq_len))

def main():
    for i, (images, midi_batch) in enumerate(val_dataloader):
        with torch.no_grad():
            images = images.to(device)
            midi_batch = midi_batch.to(device)

            output = model(images, midi_batch)
            predicted_sequence = output[0].cpu().detach().numpy().tolist()
            predicted_sequence = predicted_sequence[:32]

            midi_batch = midi_batch.cpu()
            midi_batch = midi_batch.tolist()
            midi_batch = midi_batch[0]

            predicted_midi = from_raw_to_midi(predicted_sequence)
            source_midi = from_raw_to_midi(midi_batch)

            print("Number of note sheets: ", max_midi_files)
            validate_predicted_midi(predicted_midi, source_midi, max_seq_len)

            print("Predicted:   ", predicted_midi)
            print("Source:      ", source_midi)

            print("\n\n")

if __name__ == '__main__':
    main()