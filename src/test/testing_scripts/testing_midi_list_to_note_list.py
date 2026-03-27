from src.music_program.global_variables import DELTA_TIME, NUM_NOTES, WHITE_KEYS_MIDI

note_to_index = {midi_num: i for i, midi_num in enumerate(WHITE_KEYS_MIDI)}
STOP_SIGN = -(NUM_NOTES - 1)

def create_time_series(midi_seq):
    step = DELTA_TIME[1]
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
                steps = int(delta_time // step)
                time_series.extend([note_to_index[current_note]] * steps)
                time_series.extend([STOP_SIGN])
                current_note = None

    return time_series


def time_series_to_midi_seq(time_series):
    step = DELTA_TIME[1]

    previous_note = None
    time = 0
    midi_seq_from_time_series = []

    for note in time_series:
        if note == previous_note or previous_note is None:
            previous_note = note
            time += step
            continue

        if note == STOP_SIGN:
            midi_seq_from_time_series.append((previous_note, 90, 0))
            midi_seq_from_time_series.append((previous_note, 0, time))

            time = 0
            previous_note = None

    return midi_seq_from_time_series


if "__main__" == __name__:
    midi_seq = [(64, 90, 0), (64, 0, 10080), (65, 90, 0), (65, 0, 5040), (64, 90, 0), (64, 0, 20160), (62, 90, 0), (62, 0, 30240), (62, 90, 0), (62, 0, 5040), (60, 0, 0), (60, 0, 0), (60, 0, 0), (60, 0, 0)]

    time_series = create_time_series(midi_seq)
    print(time_series)

    normalized_seq = [
        note_idx / (NUM_NOTES - 1.0)
        for note_idx in time_series
    ]
    print(normalized_seq)

    new_time_series = [
        int(round(norm_note * (NUM_NOTES - 1.0)))
        for norm_note in normalized_seq
    ]
    print(new_time_series)

    new_time_series = [
        max(STOP_SIGN, min(note, NUM_NOTES - 1))
        for note in new_time_series
    ]
    print(new_time_series)

    new_time_series = [
        WHITE_KEYS_MIDI[note_idx] if note_idx >= 0 else note_idx
        for note_idx in new_time_series
    ]
    print(new_time_series)



    midi = time_series_to_midi_seq(new_time_series)
    print(midi)

