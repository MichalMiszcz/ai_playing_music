import torch
from torchvision import transforms

from src.test.midi_series_model_index.model import ModelLSTM
from src.music_program.utils.global_variables import *
from src.utils import index_to_note_delta_time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    # transforms.RandomAffine(degrees=0, shear=2),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.5], std=[0.5])
])

model_path = 'src/model_lstm_best_index.pth'

max_seq_len = 96
max_series_len = int(max_seq_len / 2)

max_midi_files = 8192
max_midi_files_test = 1024
batch_size = 128
hidden_dim = 64
rnn_layers = 2

epochs = 100
learning_rate = 0.001
weight_decay = 0.00001
max_norm = 1.0


def time_series_to_midi_seq(time_series):
    index_dict = index_to_note_delta_time.index_to_note_delta_time_dict()
    max_index = index_to_note_delta_time.max_index()

    time_series = [
        int(round(norm_index * (max_index - 1.0)))
        for norm_index in time_series
    ]
    print(time_series)

    time_series = [
        index_dict[index]
        for index in time_series
    ]
    print(time_series)

    midi_seq_from_time_series = []

    for note, time in time_series:
        midi_seq_from_time_series.append((note, 90, 0))
        midi_seq_from_time_series.append((note, 0, time))

    return midi_seq_from_time_series


def generate_from_model(input_midi):
    model = ModelLSTM(input_dim=3, hidden_dim=hidden_dim, output_dim=1, max_seq_len=max_seq_len,
                      max_series_len=max_series_len, rnn_layers=rnn_layers)
    model = torch.compile(model)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        output = model(input_midi)
        predicted_series = output[0].cpu().detach().numpy().tolist()
        predicted_series = predicted_series[:max_series_len]
        print(predicted_series)
        predicted_series = time_series_to_midi_seq(predicted_series)

    return predicted_series


if __name__ == "__main__":
    input_sequence = [(64, 90, 0), (64, 0, 20160), (72, 90, 0), (72, 0, 20160), (72, 90, 0), (72, 0, 5040), (65, 90, 0),
                      (65, 0, 10080), (64, 90, 0), (64, 0, 5040), (69, 90, 0), (69, 0, 5040), (60, 90, 0),
                      (60, 0, 5040), (71, 90, 0), (71, 0, 5040), (71, 90, 0), (71, 0, 5040), (71, 90, 0), (71, 0, 5040),
                      (62, 90, 0), (62, 0, 5040), (71, 90, 0), (71, 0, 5040), (72, 90, 0), (72, 0, 5040), (69, 90, 0),
                      (69, 0, 20160), (60, 90, 0), (60, 0, 10080), (67, 90, 0), (67, 0, 20160), (64, 90, 0),
                      (64, 0, 10080), (65, 90, 0), (65, 0, 10080), (72, 90, 0), (72, 0, 10080), (72, 90, 0),
                      (72, 0, 20160), (60, 90, 0), (60, 0, 5040), (71, 90, 0), (71, 0, 5040), (65, 90, 0),
                      (65, 0, 10080), (67, 90, 0), (67, 0, 5040), (62, 90, 0), (62, 0, 10080), (60, 90, 0),
                      (60, 0, 5040), (65, 90, 0), (65, 0, 40320), (65, 90, 0), (65, 0, 40320)]

    input_sequence = [
        (note_to_index[note],
         velocity_to_index[velocity],
         delta_time_to_index[delta_time])
        for note, velocity, delta_time in input_sequence
    ]

    input_sequence.extend([(0, 0, 0)] * (max_seq_len - len(input_sequence)))

    input_sequence = [
        (note_idx / (NUM_NOTES - 1.0),
         velocity_idx / (NUM_VELOCITIES - 1.0),
         delta_time_idx / (NUM_DELTA_TIME - 1.0))
        for note_idx, velocity_idx, delta_time_idx in input_sequence
    ]

    input_sequence = torch.tensor(input_sequence, dtype=torch.float32).to(device)
    input_sequence = input_sequence.unsqueeze(0)

    time_series = generate_from_model(input_sequence)

    print(time_series)
