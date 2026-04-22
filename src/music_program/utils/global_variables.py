# old
# HEIGHT = 172
# WIDTH = 594

HEIGHT = 416
WIDTH = 608

WHITE_KEYS = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']

WHITE_KEYS_MIDI = [60, 62, 64, 65, 67, 69, 71, 72]
NUM_NOTES = len(WHITE_KEYS_MIDI)
STOP_SIGN = -(NUM_NOTES - 1)

VELOCITY = [0, 90]
NUM_VELOCITIES = len(VELOCITY)

# old
# DELTA_TIME = [0, 10080, 20160, 30240, 40320]
DELTA_TIME = [0, 5040, 10080, 20160, 30240, 40320] #[0, 1, 2, ..] -> [0/6, 1/6,...]
NUM_DELTA_TIME = len(DELTA_TIME)

note_to_index = {midi_num: i for i, midi_num in enumerate(WHITE_KEYS_MIDI)}
velocity_to_index = {midi_num: i for i, midi_num in enumerate(VELOCITY)}
delta_time_to_index = {midi_num: i for i, midi_num in enumerate(DELTA_TIME)}