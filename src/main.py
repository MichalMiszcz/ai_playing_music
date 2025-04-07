import subprocess
import os
import sys

midi2ly_path = 'D:/Programy/lilypond-2.24.4/bin/midi2ly.py'


def convert_midi_to_png(midi_file, folder, output_prefix="output"):

    ly_file = f"{folder}/{output_prefix}.ly"

    try:
        print("Converting MIDI to LilyPond format...")
        subprocess.run(
            [sys.executable, midi2ly_path, midi_file, "-o", ly_file],
            check=True)
    except subprocess.CalledProcessError as e:
        print("Error during MIDI to LilyPond conversion:", e)
        sys.exit(1)

    try:
        with open(ly_file, "r") as f:
            ly_content = f.read()

        header = rf'''\version "2.24.0"'
        \paper {{
        indent = 1.5\cm
        line - width = 21.0\cm
        ragged - last =  ##f
        }}
        ''' + "\n"

        layout_block = r"""
                     \layout {
                       ragged-right = ##t
                     }
                     """

        new_ly_content = header + ly_content.replace('\r\n', '\n').replace('\r', '\n') + layout_block

        with open(ly_file, "w") as f:
            f.write(new_ly_content)
    except Exception as e:
        print("Error modifying the LilyPond file:", e)
        sys.exit(1)

    try:
        print("Generating PNG from LilyPond file...")
        subprocess.run(["lilypond", "-fpng", "-o", folder, ly_file], check=True)
    except subprocess.CalledProcessError as e:
        print("Error during PNG generation:", e)
        sys.exit(1)

    png_file = f"{folder}/{output_prefix}.png"
    if os.path.exists(png_file):
        print(f"Success! PNG file created: {png_file}")
    else:
        print("PNG file was not created as expected.")


if __name__ == "__main__":
    midi_path = "music/"
    midi_file_name = "traditional-scarborough-fair.midi"
    full_path = midi_path + midi_file_name
    name, _ = os.path.splitext(full_path)
    try:
        os.mkdir(name)
    except FileExistsError:
        pass

    convert_midi_to_png(full_path, name, output_prefix="sheet_music")
