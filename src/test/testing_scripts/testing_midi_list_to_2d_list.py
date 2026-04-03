from src.music_program.utils.global_variables import *

note_to_index = {midi_num: i for i, midi_num in enumerate(WHITE_KEYS_MIDI)}
delta_time_to_index = {delta_num: i for i, delta_num in enumerate(DELTA_TIME)}

def create_time_series(midi_seq):
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
                time_series.append((note_to_index[current_note], delta_time_to_index[delta_time]))
                current_note = None

    return time_series


def time_series_to_midi_seq(time_series):
    time_series = [
        (int(round(norm_note * (NUM_NOTES - 1.0))),
        int(round(norm_delta_time * (NUM_DELTA_TIME - 1.0))))
        for norm_note, norm_delta_time in time_series
    ]
    print(time_series)

    time_series = [
        (max(0, min(WHITE_KEYS_MIDI[note], WHITE_KEYS_MIDI[NUM_NOTES - 1])),
        max(0, min(DELTA_TIME[delta_time], DELTA_TIME[NUM_DELTA_TIME - 1])))
        for note, delta_time in time_series
    ]
    print(time_series)

    midi_seq_from_time_series = []

    for note, time in time_series:
        midi_seq_from_time_series.append((note, 90, 0))
        midi_seq_from_time_series.append((note, 0, time))

    return midi_seq_from_time_series


if "__main__" == __name__:
    midi_seq = [(64, 90, 0), (64, 0, 10080), (67, 90, 0), (67, 0, 5040), (64, 90, 0), (64, 0, 20160), (62, 90, 0), (62, 0, 30240), (62, 90, 0), (62, 0, 5040), (60, 0, 0), (60, 0, 0), (60, 0, 0), (60, 0, 0)]

    # Przygotowanie do uczenia
    time_series = create_time_series(midi_seq)
    print(time_series)

    normalized_seq = [
        (note_idx / (NUM_NOTES - 1.0),
        delta_time_idx / (NUM_DELTA_TIME - 1.0))
        for note_idx, delta_time_idx in time_series
    ]
    print(normalized_seq)

    # Odczytywanie zakodowanych nut
    midi = time_series_to_midi_seq(normalized_seq)
    print(midi)

