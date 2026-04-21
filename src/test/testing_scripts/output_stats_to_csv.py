import re

from src.test.testing_scripts import parameters_to_csv

output_file = 'src/test/testing_scripts/output.txt'
log_text = open(output_file).read()

csv_path = "src/csv/parameters/autoencoder/epochs_to_csv-model_index.csv"
# version = 10

pattern = r"Epoch (\d+)/\d+, Validation Loss: ([\d\.]+)"
version_pattern = r"model_lstm_best_index_v([\d_]+)\.pth'"

def analyze_training_logs(text):
    version = None

    all_epochs_data = {}
    min_val_loss = float('inf')
    min_val_epoch = None

    for match in re.finditer(pattern, text):
        epoch = int(match.group(1))
        val_loss = float(match.group(2))

        all_epochs_data[epoch] = val_loss

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_epoch = epoch

    for match in re.finditer(version_pattern, text):
        version = match.group(1)
        version = version.replace('_', '.')
        print(f'Version: {version}')
        break

    if not all_epochs_data:
        return None

    target_epochs = [25, 50, 75, 100]
    specific_epochs_data = {}
    for ep in target_epochs:
        if ep in all_epochs_data:
            specific_epochs_data[ep] = all_epochs_data[ep]

    last_epoch = max(all_epochs_data.keys())
    last_epoch_loss = all_epochs_data[last_epoch]

    return specific_epochs_data, min_val_epoch, min_val_loss, last_epoch, last_epoch_loss, version

result = analyze_training_logs(log_text)


if result is None:
    print("Błąd: W podanym tekście nie znaleziono żadnych danych o Validation Loss.")
else:
    specific_data, best_epoch, best_loss, last_epoch, last_loss, version = result

    print("--- Validation Loss dla wybranych epok ---")
    for ep in [25, 50, 75, 100]:
        if ep in specific_data:
            print(f"Epoka {ep}: {specific_data[ep]:.6f}")
        else:
            print(f"Epoka {ep}: Brak danych (trening zakończył się wcześniej)")

    print("\n--- Najlepszy wynik (Minimum Validation Loss) ---")
    print(f"Najmniejsza wartość: {best_loss:.6f} (osiągnięta w epoce {best_epoch})")

    print("\n--- Ostatnia zarejestrowana epoka ---")
    print(f"Epoka {last_epoch}: {last_loss:.6f}")

    data_dict = {}
    specific_data_dict = {f'epoch {num}': specific_data[num] for num in specific_data}
    data_dict.update(specific_data_dict)
    data_dict['best_epoch'] = best_epoch
    data_dict['best_loss'] = best_loss
    data_dict['last_epoch'] = last_epoch
    data_dict['last_loss'] = last_loss

    print(data_dict)
    parameters_to_csv.save_data_to_df(data_dict, csv_path, f"index_v{version}")