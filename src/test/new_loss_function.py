from src.music_program.global_variables import *

from dtaidistance import dtw

def create_time_series(midi_seq, signal="on"):
    time_series = []

    for note, _, delta_time in midi_seq:
        delta_time_idx = int(delta_time * (NUM_DELTA_TIME - 1.0) + 0.5)
        delta_time_idx = max(0, min(delta_time_idx, NUM_DELTA_TIME - 1.0))
        if note is not None:
            if signal == "on":
                time_series.extend([note] * delta_time_idx)
            elif signal == "off":
                time_series.extend([note])

    return time_series

def dynamic_time_warping_score(time_series_a, time_series_b):
    max_length = max(len(time_series_a), len(time_series_b))

    time_series_a.extend([0] * (max_length - len(time_series_a)))
    time_series_b.extend([0] * (max_length - len(time_series_b)))

    distance, paths = dtw.warping_paths(time_series_a, time_series_b, use_c=False)
    best_path = dtw.best_path(paths)

    similarity_score = distance / len(best_path)

    return similarity_score

def change_delta_time_to_time(delta_seq):
    time = 0
    seq = []
    for i, item in enumerate(delta_seq):
        if i < len(delta_seq) - 1:
            time = delta_seq[i+1][2]
        seq.append([item[0], item[1], time])

    return seq

def prepare_time_series(seq, mode="on"):
    item_value = 1 if mode == "on" else 0 if mode == "off" else -1
    filtered_seq = [item for item in seq if item[1] == item_value]
    generated_time_series = create_time_series(filtered_seq, mode)
    return generated_time_series



def count_error(generated_seq, source_seq):
    generated_seq = change_delta_time_to_time(generated_seq)
    source_seq = change_delta_time_to_time(source_seq)

    generated_on_time_series = prepare_time_series(generated_seq)
    generated_off_time_series = prepare_time_series(generated_seq, "off")
    source_on_time_series = prepare_time_series(source_seq)
    source_off_time_series = prepare_time_series(source_seq, "off")

    signals_on_dtw_score = dynamic_time_warping_score(generated_on_time_series, source_on_time_series)
    signals_off_dtw_score = dynamic_time_warping_score(generated_off_time_series, source_off_time_series)

    final_score = (signals_on_dtw_score**2 + signals_off_dtw_score**2)**(1/2)
    return final_score