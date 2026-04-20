import pandas as pd

params = """
version = 1
max_seq_len = 96
max_series_len = int(max_seq_len / 2)

max_midi_files = 8192
max_midi_files_test = 1024
batch_size = 32
hidden_dim = 64
rnn_layers = 2

epochs = 100
learning_rate = 0.001
weight_decay = 0.00001
max_norm = 1.0
"""

data = {}

for line in params.strip().split("\n"):
    line = line.strip()
    if not line:
        continue

    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()

    try:
        value = eval(value, {}, data)
    except:
        pass

    data[key] = value

df = pd.DataFrame(list(data.items()), columns=["parameter", "value"])
df.to_csv(f"src/csv/parameters/autoencoder/parameters_to_csv-model_v{data['version']}.csv", index=False)