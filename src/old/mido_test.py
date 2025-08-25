import mido

def sequence_to_midi(sequence, output_midi_path):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for pitch, duration in sequence:
        if pitch > 0:  # Ignore invalid pitches
            pitch = max(0, min(127, int(pitch)))
            duration = max(0, int(duration))  # Ensure duration is in ticks
            # Note-on at the current time (initially 0, then reset after each note)
            track.append(mido.Message('note_on', note=pitch, velocity=64, time=0))
            # Note-off after the duration
            track.append(mido.Message('note_off', note=pitch, velocity=0, time=duration))

    mid.save(output_midi_path)

# Example sequence: [(pitch, duration), ...]
sequence = [(64, 480), (64, 480), (64, 480)]  # Note 64, 480 ticks each
sequence_to_midi(sequence, 'output.mid')