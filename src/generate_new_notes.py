import music21
import random
import os

WHITE_KEYS = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']

NUM_SONGS = 3
MEASURES_PER_SONG = 4
NOTES_PER_MEASURE = 4
TEMPO = 120

output_folder_path = "generated_songs"


def generate_random_song(song_number):
    song = music21.stream.Score()
    part = music21.stream.Part()

    part.append(music21.tempo.MetronomeMark(number=TEMPO))
    part.append(music21.key.KeySignature(0))

    part.append(music21.meter.TimeSignature('4/4'))

    for _ in range(MEASURES_PER_SONG):
        measure_duration = 0.0
        while measure_duration < 4.0:
            duration = 1.0
            if measure_duration + duration > 4.0:
                duration = 4.0 - measure_duration
            if duration <= 0:
                break

            pitch = random.choice(WHITE_KEYS)
            note = music21.note.Note(pitch, quarterLength=duration)
            part.append(note)
            measure_duration += duration

    song.append(part)

    output_dir = f'{output_folder_path}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'song_{song_number}.mid')
    song.write('midi', output_path)
    print(f'Saved song {song_number} to {output_path}')


def main():
    for i in range(1, NUM_SONGS + 1):
        generate_random_song(i)


if __name__ == '__main__':
    main()