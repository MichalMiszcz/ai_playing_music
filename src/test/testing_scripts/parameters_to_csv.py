import pandas as pd
import os

csv_path = "src/csv/parameters/autoencoder/parameters_to_csv-model.csv"

params_string = """
version = 3
max_seq_len = 96  
max_series_len = int(max_seq_len / 2)  
vocab_size = max_index() + 1   
  
max_midi_files=8192  
max_midi_files_test=1024  
batch_size=32  
hidden_dim=64  
embedding_dim=64  
rnn_layers=2  
  
epochs=100  
learning_rate = 0.001  
weight_decay = 0.00001  
max_norm=1.0

teacher_epochs=3
"""

def extract_params_from_string(params):
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

    return data

def save_data_to_df(data, path, column_name):
    new_df = pd.DataFrame(list(data.items()), columns=["parameter", column_name])
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)

        # merge on index (parameter names)
        df = df.merge(new_df, on="parameter", how='outer')
    else:
        df = new_df

    df.to_csv(path, index=False)

if __name__ == '__main__':
    params_data = extract_params_from_string(params_string)
    save_data_to_df(params_data, csv_path, f"index_v{params_data["version"]}")