from src.music_program.utils.global_variables import *

def max_index():
    max_index = (NUM_DELTA_TIME - 1) + (NUM_DELTA_TIME - 1) * (NUM_NOTES - 1)
    return max_index

def make_indexes(note_idx, delta_time_idx):
    index = delta_time_idx + (NUM_DELTA_TIME - 1) * note_idx
    return index

def index_to_note_delta_time_dict():
    WHITE_KEYS_MIDI_temp = WHITE_KEYS_MIDI.copy()
    DELTA_TIME_temp = DELTA_TIME.copy()

    # DELTA_TIME_temp.remove(0)

    index_dict = {
        make_indexes(note_to_index[note], delta_time_to_index[dt]): (note, dt)
        for note in WHITE_KEYS_MIDI_temp
        for dt in DELTA_TIME_temp
    }

    return index_dict