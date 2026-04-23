from src.music_program.utils.global_variables import *

index_to_delta_time = {i: midi_num for i, midi_num in enumerate(DELTA_TIME)}


def normalize_3d_midi_sequence(midi_seq):
    """
    Method to normalize three-dimensional midi sequence
    :param midi_seq: sequence that stores information from midi. Structure of sequence: [(note, velocity, delta_time), ...]
    :return: normalized three-dimensional midi sequence
    """

    normalized_midi_seq = [
        (note_idx / (NUM_NOTES - 1.0),
         velocity_idx / (NUM_VELOCITIES - 1.0),
         index_to_delta_time[delta_time_idx] / MAX_DELTA_TIME)
        for note_idx, velocity_idx, delta_time_idx in midi_seq
    ]

    return normalized_midi_seq


def create_2d_time_series(midi_seq, max_series_len):
    """
    Method to create two-dimensional time series of notes
    :param midi_seq: sequence that stores information from midi. Structure of sequence: [(note, velocity, delta_time), ...]
    :param max_series_len: maximum length of time series
    :return: two-dimensional time series of notes. Structure of sequence: [(note, delta_time), ...]
    """
    time_series = []
    current_note = None

    for row in midi_seq:
        note = row[0]
        velocity = row[1]
        delta_time = row[2]

        if velocity > 0:
            current_note = note
        else:
            if current_note is not None:
                time_series.append((note, delta_time))
                current_note = None

    if len(time_series) < max_series_len:
        time_series.extend([(0, 0)] * (max_series_len - len(time_series)))

    return time_series


if __name__ == "__main__":
    midi_seq = [(1, 1, 0), (1, 0, 1), (1, 1, 0), (1, 0, 3), (3, 1, 0), (3, 0, 5)]
    norm_midi_seq = normalize_3d_midi_sequence(midi_seq)
    print(norm_midi_seq)

    norm_midi_seq_2d = create_2d_time_series(norm_midi_seq, max_series_len=3)
    print(norm_midi_seq_2d)
