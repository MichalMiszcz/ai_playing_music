import subprocess
import random

import music21
import os
import sys
import mido
from PIL import Image

from os import mkdir
from src.utils import instances

MUSESCORE_EXECUTABLE = "musescore4"

def preprocess_midi(midi_file, output_midi_file, max_duration=10.0, fixed_bpm=120, add_track_name=True, previous_track_name=''):
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

            if msg.type == 'pitchwheel':
                continue

            if msg.is_meta and msg.type == 'set_tempo':
                continue

            if add_track_name:
                if msg.is_meta and msg.type == 'track_name':
                    if add_track_name and track.name == previous_track_name:
                        new_track.append(mido.MetaMessage('track_name', name='Piano right'))
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
        command = [musescore_path, "-r 72", midi_file, "-o", output_file]
        print("Converting MIDI to sheet music using MuseScore...")
        subprocess.run(command, check=True)
        print(f"Success! Sheet music generated: {output_file}")
    except subprocess.CalledProcessError as e:
        print("Error during conversion:")
        print(e)
        sys.exit(1)

def convert_xml_to_sheet(xml_file, output_file, musescore_path="MuseScore4.exe"):
    if not os.path.exists(xml_file):
        print(f"Error: MusicXML file '{xml_file}' not found.")
        sys.exit(1)

    try:
        resolution = random.randint(72, 80)
        print("Resolution:", resolution)
        command = [musescore_path, f"-r {resolution}", xml_file, "-o", output_file]
        print("Converting MusicXML to sheet music using MuseScore...")
        subprocess.run(command, check=True)
        print(f"Success! Sheet music generated: {output_file}")
    except subprocess.CalledProcessError as e:
        print("Error during conversion:")
        print(e)
        sys.exit(1)

def process_midi(midi_folder_path, processed_folder_path, max_duration=10.0, fixed_bpm=60, add_track_name=True):
    for root, dirs, files in os.walk(midi_folder_path):
        # for folder in dirs:
        # active_midi_folder = midi_folder_path + "/" + folder
        active_midi_folder = midi_folder_path
        # active_processed_midi_folder = processed_folder_path + "/" + folder
        active_processed_midi_folder = processed_folder_path

        if not os.path.exists(active_processed_midi_folder):
            mkdir(active_processed_midi_folder)

        # for file in os.listdir(os.path.join(root, folder)):
        for file in os.listdir(root):
            input_midi_file = active_midi_folder + "/" + file
            output_midi_file = active_processed_midi_folder + "/" + file
            print(f"Processing MIDI: {output_midi_file}")

            preprocess_midi(input_midi_file, output_midi_file, max_duration=max_duration, fixed_bpm=fixed_bpm, add_track_name=add_track_name)

def midi2jpg(midi_folder_path, image_folder_path, mode='midi', instances_count = 1):
    for root, dirs, files in os.walk(midi_folder_path):
        # for folder in dirs:
        # active_midi_folder = midi_folder_path + "/" + folder
        # active_image_folder = image_folder_path + "/" + folder
        active_midi_folder = midi_folder_path
        active_image_folder = image_folder_path

        if not os.path.exists(active_image_folder):
            mkdir(active_image_folder)

        instance_num, lock_socket = instances.get_instance_id(max_instances=instances_count)

        all_files = sorted(os.listdir(root))

        # for file in os.listdir(os.path.join(root, folder)):
        for i, file in enumerate(all_files):
            num = file[5:]
            num = num[:-9]
            num = int(num)
            if i % instances_count == instance_num - 1:
                input_midi_file = active_midi_folder + "/" + file
                file_base = os.path.splitext(file)[0]

                single_image_folder = active_image_folder + "/" + file_base

                if not os.path.exists(single_image_folder):
                    mkdir(single_image_folder)
                else:
                    continue

                print(single_image_folder)

                output_sheet_file = single_image_folder + "/" + file_base + ".png"
                output_sheet_file_to_change = single_image_folder + "/" + file_base + "-1" + ".png"

                # print(f"Processing MIDI: {input_midi_file}")
                # print(f"Generating sheet music: {output_sheet_file}")

                if mode == 'midi':
                    convert_midi_to_sheet(input_midi_file, output_sheet_file, MUSESCORE_EXECUTABLE)
                elif mode == 'xml':
                    convert_xml_to_sheet(input_midi_file, output_sheet_file, MUSESCORE_EXECUTABLE)

                img = Image.open(output_sheet_file_to_change)

                if mode == 'midi':
                    crop_values = (0, 0, img.width - 10, round(3 * img.height / 5, 0) - 115)
                else:
                    crop_y = random.randint(0, 25)
                    crop_x = random.randint(0, 10)

                    left = 10 - crop_x
                    right = img.width - crop_x
                    top = 100 + crop_y
                    bottom = round(3 * img.height / 5, 0) - 15 + crop_y

                    crop_values = (left, top, right, bottom)

                cropped = img.crop(crop_values)
                cropped.save(output_sheet_file_to_change)


if __name__ == "__main__":
    # midi_raw_folder_path = "generated_songs_raw"
    midi_raw_folder_path = "../src/all_data/generated/generated_complex_midi_test/my_midi_files"
    xml_raw_folder_path = "../src/all_data/generated/generated_complex_midi_test/my_xml_files"
    # processed_folder_path = "data/processed_midi"
    # processed_folder_path = "generated_songs_processed"
    processed_folder_path = "../src/all_data/generated/generated_complex_midi_processed/my_midi_files"
    # image_folder_path = "data/images"
    # image_folder_path = "my_images/my_midi_images"
    image_folder_path = "../src/all_data/generated/my_complex_images_test/my_midi_images/my_midi_files"

    # process_midi(midi_raw_folder_path, processed_folder_path, max_duration=24.0, fixed_bpm=120, add_track_name=True)
    # midi2jpg(processed_folder_path, image_folder_path)
    midi2jpg(xml_raw_folder_path, image_folder_path, 'xml', instances_count=4)