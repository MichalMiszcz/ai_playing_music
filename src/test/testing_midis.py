import mido

def preprocess_midi(midi_file, max_duration=10.0, fixed_bpm=120):
    try:
        mid = mido.MidiFile(midi_file)
    except Exception as e:
        print(f"Error reading MIDI file '{midi_file}': {e}")
        return

    # new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

    print("==== MIDI FILE INFO ====")
    print(f"Type: {mid.type}")
    print(f"Number of Tracks: {len(mid.tracks)}")
    print(f"Ticks per Beat: {mid.ticks_per_beat}")
    print(f"Length (seconds): {mid.length:.2f}")

    # Print information for each track
    print("\n==== TRACKS ====")
    for i, track in enumerate(mid.tracks):
        print(f"\nTrack {i}: {track.name}")
        for msg in track:
            print(msg)

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
    # midi_example = "my_data/my_midi_files/simple_piano_02.mid"
    # midi_example = "../src/generated_songs_processed/my_midi_files/song_2.mid"
    midi_example = "src/all_data/generated/generated_songs_processed_test/my_midi_files/kotek.mid"
    preprocess_midi(midi_example)

    midi_example = "src/all_data/generated/generated_songs_processed_test/my_midi_files/song_1.mid"
    preprocess_midi(midi_example)

