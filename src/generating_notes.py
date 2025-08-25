import subprocess
import os
import sys
import mido

from os import mkdir

MUSESCORE_EXECUTABLE = "musescore4"

def preprocess_midi(midi_file, output_midi_file, max_duration=10.0, fixed_bpm=120):
    try:
        mid = mido.MidiFile(midi_file)
    except Exception as e:
        print(f"Error reading MIDI file '{midi_file}': {e}")
        return

    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

    tempo = mido.bpm2tempo(fixed_bpm)

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        new_mid.tracks.append(new_track)

        new_track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

        current_time = 0.0

        for msg in track:
            if msg.time > 0:
                current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)

            if current_time > max_duration:
                break

            if msg.is_meta and msg.type == 'set_tempo':
                continue

            new_track.append(msg)

    try:
        new_mid.save(output_midi_file)
    except Exception as e:
        print(f"Error saving processed MIDI file '{output_midi_file}': {e}")
        return

def convert_midi_to_sheet(midi_file, output_file, musescore_path="MuseScore4.exe"):
    if not os.path.exists(midi_file):
        print(f"Error: MIDI file '{midi_file}' not found.")
        sys.exit(1)

    try:
        command = [musescore_path, midi_file, "-o", output_file]
        print("Converting MIDI to sheet music using MuseScore...")
        subprocess.run(command, check=True)
        print(f"Success! Sheet music generated: {output_file}")
    except subprocess.CalledProcessError as e:
        print("Error during conversion:")
        print(e)
        sys.exit(1)

def process_midi(midi_folder_path, processed_folder_path, max_duration=10.0, fixed_bpm=60):
    for root, dirs, files in os.walk(midi_folder_path):
        for folder in dirs:
            active_midi_folder = midi_folder_path + "/" + folder
            active_processed_midi_folder = processed_folder_path + "/" + folder

            if not os.path.exists(active_processed_midi_folder):
                mkdir(active_processed_midi_folder)

            for file in os.listdir(os.path.join(root, folder)):
                input_midi_file = active_midi_folder + "/" + file
                output_midi_file = active_processed_midi_folder + "/" + file
                print(f"Processing MIDI: {output_midi_file}")

                preprocess_midi(input_midi_file, output_midi_file, max_duration=max_duration, fixed_bpm=fixed_bpm)

def midi2jpg(midi_folder_path, image_folder_path):
    for root, dirs, files in os.walk(midi_folder_path):
        for folder in dirs:
            active_midi_folder = midi_folder_path + "/" + folder
            active_image_folder = image_folder_path + "/" + folder

            if not os.path.exists(active_image_folder):
                mkdir(active_image_folder)

            for file in os.listdir(os.path.join(root, folder)):
                input_midi_file = active_midi_folder + "/" + file
                file_base = os.path.splitext(file)[0]

                single_image_folder = active_image_folder + "/" + file_base

                if not os.path.exists(single_image_folder):
                    mkdir(single_image_folder)

                print(single_image_folder)

                output_sheet_file = single_image_folder + "/" + file_base + ".png"
                print(f"Processing MIDI: {input_midi_file}")
                print(f"Generating sheet music: {output_sheet_file}")

                convert_midi_to_sheet(input_midi_file, output_sheet_file, MUSESCORE_EXECUTABLE)


if __name__ == "__main__":
    # midi_raw_folder_path = "data/midi"
    # processed_folder_path = "data/processed_midi"
    processed_folder_path = "my_data"
    # image_folder_path = "data/images"
    image_folder_path = "my_images/my_midi_images"

    # process_midi(midi_raw_folder_path, processed_folder_path, max_duration=16.0, fixed_bpm=60)
    midi2jpg(processed_folder_path, image_folder_path)