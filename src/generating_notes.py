import subprocess
import os
import sys
import mido

from os import mkdir

MUSESCORE_EXECUTABLE = "musescore4"

def preprocess_midi(midi_file, output_midi_file):
    """
    Preprocess MIDI file to remove or standardize velocity values.
    """
    mid = mido.MidiFile(midi_file)
    for track in mid.tracks:
        track[:] = [msg for msg in track if msg.type != 'set_tempo']
    mid.save(output_midi_file)

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

    # temp_midi_file = midi_file.replace(".mid", "_processed.mid")
    # preprocess_midi(midi_file, temp_midi_file)

    try:
        # Construct the command to call MuseScore
        # Example: MuseScore4.exe input.mid -o output.png
        command = [musescore_path, midi_file, "-o", output_file]
        print("Converting MIDI to sheet music using MuseScore...")
        subprocess.run(command, check=True)
        print(f"Success! Sheet music generated: {output_file}")
        # os.remove(temp_midi_file)
    except subprocess.CalledProcessError as e:
        # os.remove(temp_midi_file)
        print("Error during conversion:")
        print(e)
        sys.exit(1)

# MIDI processed
def process_midi():
    midi_folder_path = "data/midi"
    processed_folder_path = "data/processed_midi"

    for root, dirs, files in os.walk(midi_folder_path):
        for folder in dirs:
            active_midi_folder = midi_folder_path + "/" + folder
            active_processed_midi_folder = processed_folder_path + "/" + folder

            path_exist = os.path.exists(active_processed_midi_folder)

            if not path_exist:
                mkdir(active_processed_midi_folder)

            for file in os.listdir(os.path.join(root, folder)):
                input_midi_file = active_midi_folder + "/" + file
                output_sheet_file = active_processed_midi_folder + "/" + file
                print(output_sheet_file)

                preprocess_midi(input_midi_file, output_sheet_file)

# MIDI to MP3 can be converted the same way
# MIDI to PDF
def midi2pdf():
    midi_folder_path = "data/processed_midi"
    pdf_folder_path = "data/pdf"

    print("Converting MIDI to PDF...")

    for root, dirs, files in os.walk(midi_folder_path):
        for folder in dirs:
            active_midi_folder = midi_folder_path + "/" + folder
            active_pdf_folder = pdf_folder_path + "/" + folder

            path_exist = os.path.exists(active_pdf_folder)

            if not path_exist:
                mkdir(active_pdf_folder)

            for file in os.listdir(os.path.join(root, folder)):
                input_midi_file = active_midi_folder + "/" + file

                file = os.path.splitext(file)[0]
                output_sheet_file = active_pdf_folder + "/" + file + ".pdf"
                print(output_sheet_file)

                convert_midi_to_sheet(input_midi_file, output_sheet_file, MUSESCORE_EXECUTABLE)

# JPG from MIDI
def midi2jpg():
    midi_folder_path = "data/processed_midi"
    pdf_folder_path = "data/images"

    for root, dirs, files in os.walk(midi_folder_path):
        for folder in dirs:
            active_midi_folder = midi_folder_path + "/" + folder
            active_pdf_folder = pdf_folder_path + "/" + folder

            path_exist = os.path.exists(active_pdf_folder)

            if not path_exist:
                mkdir(active_pdf_folder)

            for file in os.listdir(os.path.join(root, folder)):
                input_midi_file = active_midi_folder + "/" + file

                file = os.path.splitext(file)[0]
                output_sheet_folder = active_pdf_folder + "/" + file
                if not os.path.exists(output_sheet_folder):
                    mkdir(output_sheet_folder)

                output_sheet_file = output_sheet_folder + "/" + file + ".png"
                print(output_sheet_file)

                convert_midi_to_sheet(input_midi_file, output_sheet_file, MUSESCORE_EXECUTABLE)

if __name__ == "__main__":
    # process_midi()
    # midi2pdf()
    # midi2jpg()
    pass