import pandas as pd

from src.music_program.utils.global_variables import *
from src.test.validating.accuracy import *


midi_columns = ['midi_note', 'velocity', 'delta_time']


def time_series_to_midi_seq(time_series, mode="old"):
    def round_to_list(value, target_list):
        return min(target_list, key=lambda x: abs(x - value))

    if mode == "old":
        time_series = [
            (int(round(norm_note * (NUM_NOTES - 1.0))),
             delta_time_to_index[int(round_to_list((norm_delta_time * MAX_DELTA_TIME), DELTA_TIME))])
            for norm_note, norm_delta_time in time_series
        ]

    time_series = [
        (max(0, min(WHITE_KEYS_MIDI[note], WHITE_KEYS_MIDI[NUM_NOTES - 1])),
         max(0, min(DELTA_TIME[delta_time], DELTA_TIME[NUM_DELTA_TIME - 1])))
        for note, delta_time in time_series
    ]

    midi_seq_from_time_series = []

    for note, time in time_series:
        midi_seq_from_time_series.append((note, 90, 0))
        midi_seq_from_time_series.append((note, 0, time))

    return midi_seq_from_time_series


def validate_predicted_midi(df_predicted: pd.DataFrame, df_source: pd.DataFrame):
    stats_df = pd.DataFrame()

    df_predicted['velocity_normalized'] = df_predicted[midi_columns[1]] / 90
    df_source['velocity_normalized'] = df_source[midi_columns[1]] / 90

    # df_predicted['delta_time_s_normalized'] = df_predicted['delta_time_s'] / 5040
    # df_source['delta_time_s_normalized'] = df_source['delta_time_s'] / 5040

    # dtw_score = dynamic_time_warping_score_multi_col(df_predicted, df_source, [midi_columns[0], 'delta_time_s_normalized'])
    dtw_score = dynamic_time_warping_score(df_predicted, df_source)
    levenstein = edit_distance_multi_col(df_predicted, df_source,
                                         [midi_columns[0], 'velocity_normalized', 'delta_time_s'])
    frechet = discrete_frechet(df_predicted, df_source, [midi_columns[0], 'velocity_normalized', 'time'])

    stats_df['DTW score'] = [dtw_score]
    stats_df['Levenstein score'] = [levenstein]
    stats_df['Frechet score'] = [frechet]

    return stats_df
    # return stats['midi_note'], stats['velocity'], stats['delta_time']


def midi_to_df(midi_seq):
    df_midi = pd.DataFrame(midi_seq, columns=midi_columns)
    df_midi['delta_time_s'] = df_midi['delta_time'] / (10080 * 2)  # quarter note = 0.5s
    df_midi['time'] = df_midi['delta_time_s'].cumsum()

    return df_midi


def calculate_measures(predicted_sequence, source_sequence):
    df_predicted_midi = midi_to_df(predicted_sequence)
    df_source_midi = midi_to_df(source_sequence)

    df_results = validate_predicted_midi(df_predicted_midi, df_source_midi)

    print("Predicted:   ", predicted_sequence)
    print("Source:      ", source_sequence)
    print()

    return df_results