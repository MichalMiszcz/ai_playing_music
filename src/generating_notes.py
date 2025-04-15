import subprocess
import os
import sys


def convert_midi_to_sheet(midi_file, output_file, musescore_path="MuseScore4.exe"):
    """
    Converts a MIDI file to a sheet music image using MuseScore's CLI.

    Parameters:
        midi_file (str): Path to the input MIDI file.
        output_file (str): Path for the output file (should end in .pdf, .png, etc.).
        musescore_path (str): Path to the MuseScore executable.
    """
    # Check if input file exists
    if not os.path.exists(midi_file):
        print(f"Error: MIDI file '{midi_file}' not found.")
        sys.exit(1)

    try:
        # Construct the command to call MuseScore
        # Example: MuseScore4.exe input.mid -o output.png
        command = [musescore_path, midi_file, "-o", output_file]
        print("Converting MIDI to sheet music using MuseScore...")
        subprocess.run(command, check=True)
        print(f"Success! Sheet music generated: {output_file}")
    except subprocess.CalledProcessError as e:
        print("Error during conversion:")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    # Replace with the path to your MIDI file and desired output file name.
    midi_file_path = "music/pachelbel-johann-cannon.midi"
    output_sheet_path = "sheet_music.png"

    # If MuseScore is not added to your PATH, provide its full path:
    # musescore_executable = "C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe"
    musescore_executable = "musescore4"

    convert_midi_to_sheet(midi_file_path, output_sheet_path, musescore_executable)
