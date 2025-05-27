import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import mido

# Define the CNNRNNModel (must match the training script)
class CNNRNNModel(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=512, output_dim=2, rnn_layers=2):
        super(CNNRNNModel, self).__init__()
        # CNN for feature extraction
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Identity()
        self.cnn.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # RNN for sequence generation
        self.rnn = nn.LSTM(input_size=output_dim, hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

        # Projection for initial hidden state
        self.proj_h = nn.Linear(hidden_dim, hidden_dim * rnn_layers)
        self.proj_c = nn.Linear(hidden_dim, hidden_dim * rnn_layers)

    def forward(self, x, target=None):
        batch_size = x.size(0)
        features = self.cnn(x).view(batch_size, -1)  # [batch_size, hidden_dim]

        # Initialize RNN hidden state
        h0 = self.proj_h(features).view(batch_size, self.rnn.num_layers, -1).transpose(0, 1).contiguous()
        c0 = self.proj_c(features).view(batch_size, self.rnn.num_layers, -1).transpose(0, 1).contiguous()

        if target is not None:
            # Teacher forcing during training
            input_seq = torch.cat([torch.zeros(batch_size, 1, 2).to(x.device), target[:, :-1, :]], dim=1)
            output, _ = self.rnn(input_seq, (h0, c0))
            output = self.linear(output)
            return output
        else:
            # Autoregressive generation
            output_seq = []
            input_note = torch.zeros(batch_size, 1, 2).to(x.device)
            hidden = (h0, c0)
            for _ in range(100):  # Fixed sequence length
                output, hidden = self.rnn(input_note, hidden)
                output = self.linear(output)
                output_seq.append(output)
                input_note = output
            return torch.cat(output_seq, dim=1)

# Function to convert sequence to MIDI
def sequence_to_midi(sequence, output_midi_path):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for pitch, duration in sequence:
        if pitch > 0:  # Ignore padding
            pitch = max(0, min(127, int(pitch)))  # Clamp pitch to valid MIDI range
            duration = max(0, int(duration))  # Convert duration to ticks
            track.append(mido.Message('note_on', note=pitch, velocity=64, time=0))
            track.append(mido.Message('note_off', note=pitch, velocity=0, time=duration))

    mid.save(output_midi_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model
model = CNNRNNModel(input_channels=3, hidden_dim=512, output_dim=2)
model.to(device)

# Load the saved model
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.eval()
print("Model loaded from model.pth")

# Image transformation (must match training)
image_transform = transforms.Compose([
    transforms.Resize((496, 701)),  # Match training resize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess a single image
image_path = "data/images/albeniz/alb_esp1/alb_esp1-1.png"  # Replace with your image path
image = Image.open(image_path).convert('RGB')
image = image_transform(image).unsqueeze(0)  # Add batch dimension: [1, 3, 496, 701]
image = image.to(device)

# Generate MIDI
with torch.no_grad():
    output = model(image)  # No target during inference
    predicted_sequence = output[0].cpu().detach().numpy().tolist()
    print(predicted_sequence)
    # predicted_sequence = [(int(abs(x) * 1000000), int(abs(y) * 1000000)) for x, y in predicted_sequence]
    # predicted_sequence = [(int(abs(x) * 500), int(abs(y) * 8)) for x, y in predicted_sequence]
    predicted_sequence = [(int(abs(x) * 127), int(abs(y) * 150)) for x, y in predicted_sequence]
    # predicted_sequence = [(int(abs(x)), int(abs(y))) for x, y in predicted_sequence]
    # predicted_sequence = [(int(p * 127 + 0.5), int(d * 10)) for p, d in predicted_sequence]
    print(predicted_sequence)
    sequence_to_midi(predicted_sequence, "generated_output_from_main_2.mid")
    print("MIDI file generated: generated_output.mid")


