import os

import mido

def preprocess_midi(midi_folder):
    midis_list = []

    for root, dirs, files in os.walk(midi_folder):
        for folder in dirs:
            dir_path = os.path.join(root, folder)
            for midi_file in os.listdir(dir_path):
                try:
                    mid = mido.MidiFile(os.path.join(dir_path, midi_file))
                except Exception as e:
                    print(f"Error reading MIDI file '{midi_file}': {e}")
                    return

                track_dict = {}
                for i, track in enumerate(mid.tracks):
                    # print(f"\nTrack {i}: {track.name}")
                    if i == 1 or i == 2:
                        track_dict[f"{midi_file}_{i}"] = track.name
                    # for msg in track:
                        # print(msg)

                midis_list.append(track_dict)

    print(midis_list)

    distinct_names = []
    seen = set()

    looked = ['Piano right', 'Piano left', 'Right', 'Left']
    count = 0
    count_good = 0

    for data in midis_list:
        for name in data.values():
            if name not in looked:
                count += 1
            else:
                count_good += 1
                # distinct_names.append(name)
                # seen.add(name)

    print(count, count_good)


    # tempo = mido.bpm2tempo(fixed_bpm)

    # for track in mid.tracks:
    #     new_track = mido.MidiTrack()
    #     new_mid.tracks.append(new_track)
    #
    #     new_track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    #
    #     current_time = 0.0
    #
    #     for msg in track:
    #         if msg.time > 0:
    #             current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
    #
    #         if current_time > max_duration:
    #             break
    #
    #         if msg.is_meta and msg.type == 'set_tempo':
    #             continue
    #
    #         new_track.append(msg)


if __name__ == "__main__":
    midi_example = "data/processed_midi"

    preprocess_midi(midi_example)

