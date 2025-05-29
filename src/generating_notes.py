import subprocess
import os
import sys
import mido

from os import mkdir

MUSESCORE_EXECUTABLE = "musescore4"

def preprocess_midi(midi_file, output_midi_file, max_duration=10.0, fixed_bpm=120):
    """
    Preprocess MIDI file to set a fixed tempo and trim to a specified duration (in seconds).

    Parameters:
        midi_file (str): Path to the input MIDI file.
        output_midi_file (str): Path for the output MIDI file.
        max_duration (float): Maximum duration to keep (in seconds).
        fixed_bpm (int): Fixed tempo in beats per minute (default: 120 BPM).
    """
    try:
        mid = mido.MidiFile(midi_file)
    except Exception as e:
        print(f"Error reading MIDI file '{midi_file}': {e}")
        return

    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

    # Convert BPM to microseconds per beat for set_tempo meta-message
    tempo = mido.bpm2tempo(fixed_bpm)

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        new_mid.tracks.append(new_track)

        # Add a set_tempo meta-message at the start of each track
        new_track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

        current_time = 0.0  # Time in seconds

        for msg in track:
            # Convert delta time (ticks) to seconds using the fixed tempo
            if msg.time > 0:
                current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)

            # Stop adding messages if we exceed max_duration
            if current_time > max_duration:
                break

            # Skip existing tempo changes to enforce the fixed tempo
            if msg.is_meta and msg.type == 'set_tempo':
                continue

            # Add the message to the new track
            new_track.append(msg)

        # Ensure the track has an end_of_track meta-message
        # if not any(msg.is_meta and msg.type == 'end_of_track' for msg in new_track):
        #     new_track.append(mido.MetaMessage('end_of_track', time=0))

    try:
        new_mid.save(output_midi_file)
    except Exception as e:
        print(f"Error saving processed MIDI file '{output_midi_file}': {e}")
        return

def convert_midi_to_sheet(midi_file, output_file, musescore_path="MuseScore4.exe"):
    """
    Converts a MIDI file to a sheet music image using MuseScore's CLI.

    Parameters:
        midi_file (str): Path to the input MIDI file.
        output_file (str): Path for the output file (should end in .pdf, .png, etc.).
        musescore_path (str): Path to the MuseScore executable.
    """
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

                # Convert the processed MIDI to PNG
                convert_midi_to_sheet(input_midi_file, output_sheet_file, MUSESCORE_EXECUTABLE)

if __name__ == "__main__":
    midi_raw_folder_path = "data/midi"
    processed_folder_path = "data/processed_midi"
    image_folder_path = "data/images"

    process_midi(midi_raw_folder_path, processed_folder_path, max_duration=16.0, fixed_bpm=60)
    midi2jpg(processed_folder_path, image_folder_path)