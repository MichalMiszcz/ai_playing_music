import torch
from torch import nn
from torchvision.transforms import v2
from PIL import Image

from src.music_program.model.cnnrnn_model_7_1 import CNNRNNModel
from src.music_program.dataset.music_image_dataset_7_1 import MusicImageDataset

from torchvision.transforms.functional import to_pil_image

max_seq_len = 96
max_series_len = int(max_seq_len / 2)

model_path = "src/_models/image_to_midi/model_best_v310_big_kernel.pth"
batch_size = 256
hidden_dim = 1024
rnn_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TruncatedModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        layers = list(original_model.children())
        self.layers = nn.ModuleList(layers[:1])


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


model = CNNRNNModel(input_channels=1, hidden_dim=hidden_dim, output_dim=2, max_seq_len=max_seq_len,
                    max_series_len=max_series_len, rnn_layers=rnn_layers)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

layers = list(model.children())
# print(layers)

layers = nn.ModuleList(layers[:1])
# print(layers)

truncated = TruncatedModel(model)
image_path = 'src/all_data/generated/my_complex_images/my_midi_images/my_midi_files/song_1/song_1-1.png'
image_path_1 = 'src/all_data/generated/my_complex_images/my_midi_images/my_midi_files/song_3/song_3-1.png'

img_paths = [image_path, image_path_1]

image_transform = v2.Compose([
    v2.Resize((512, 512)),
    # v2.RandomAffine(degrees=1, shear=0),
    # v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # v2.RandomInvert(p=1.0),
    # v2.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0)
])

vectors = []
for img in img_paths:
    image = Image.open(img).convert('RGB')
    image = image_transform(image)

    image_to_show = to_pil_image(image)
    image_to_show.show("Modified image")

    image = image.reshape(1, 3, 512, 512).to(device)

    output = truncated(image)
    print(output)
    vectors.append(output)

result = torch.norm(vectors[0] - vectors[1])
print(result)




# self.layers = nn.ModuleList(layers[:num_layers])

# for name, param in model.named_parameters():
#     print(name)
#     print(param.data)
#     print()
