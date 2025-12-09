import music21
import random
import os
from src.music_program.global_variables import WHITE_KEYS
from src.utils import instances

NUM_SONGS = 1
NOTES_PER_MEASURE = 4.0
TEMPO = 120

output_folder_path = "../src/all_data/generated/generated_complex_midi_test"

def generate_random_song(song_number):
    song = music21.stream.Score()
    part = music21.stream.Part()

    part.append(music21.tempo.MetronomeMark(number=TEMPO))
    part.append(music21.key.KeySignature(0))

    part.append(music21.meter.TimeSignature('4/4'))

    measures_per_song = random.choice([4, 8, 12])

    for i in range(measures_per_song):
        measure = music21.stream.Measure()
        measure.number = i + 1

        if i % 4 == 0:
            measure.append(music21.layout.SystemLayout(isNew=True))

        options = [1.0, 2.0, 4.0, 0.5]
        selection = []

        previous_choice = 0.0

        while sum(selection) < NOTES_PER_MEASURE:
            remaining = NOTES_PER_MEASURE - sum(selection)
            valid_options = [x for x in options if x <= round(remaining, 2)]

            if not valid_options:
                break

            while True:
                choice = random.choice(valid_options)
                if abs(choice - previous_choice) == 0:
                    break

                if abs(choice - previous_choice) in valid_options:
                    break

                if NOTES_PER_MEASURE - sum(selection) == choice:
                    break

            selection.append(choice)
            previous_choice = choice

        for duration in selection:
            pitch = random.choice(WHITE_KEYS)
            note = music21.note.Note(pitch, quarterLength=duration)
            measure.append(note)

        part.append(measure)

    song.append(part)

    output_dir = f'{output_folder_path}/my_midi_files'
    output_dir_xml = f'{output_folder_path}/my_xml_files'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'song_{song_number}.mid')
    output_path_xml = os.path.join(output_dir_xml, f'song_{song_number}.musicxml')
    song.write('midi', output_path)
    song.write('musicxml', output_path_xml)
    print(f'Saved song {song_number} to {output_path}')


def main():
    max_instances = 1

    instance_num, lock_socket = instances.get_instance_id(max_instances=max_instances)

    for i in range(int((instance_num-1) * NUM_SONGS/max_instances) + 1, int(instance_num * NUM_SONGS/max_instances) + 1):
        generate_random_song(i)

if __name__ == '__main__':
    main()