import subprocess
import os
import sys
import mingus.extra.lilypond as lilypond_extra  # For potential LilyPond modifications

midi2ly_path = 'D:/Programy/lilypond-2.24.4/bin/midi2ly.py'


def convert_midi_to_png(midi_file, folder, output_prefix="output"):
    """
    Converts a MIDI file to a PNG image of sheet music using midi2ly and LilyPond.

    Parameters:
        midi_file (str): Path to the input MIDI file.
        output_prefix (str): Prefix for output files (LilyPond and PNG).
        :param folder:
    """
    # Define the output LilyPond (.ly) file path.
    ly_file = f"{folder}/{output_prefix}.ly"

    try:
        # Step 1: Convert MIDI to LilyPond file using midi2ly.
        # The -o flag specifies the output file.
        print("Converting MIDI to LilyPond format...")
        # subprocess.run(["midi2ly", midi_file, "-o", ly_file], check=True)
        subprocess.run(
            [sys.executable, midi2ly_path, midi_file, "-o", ly_file],
            check=True)
    except subprocess.CalledProcessError as e:
        print("Error during MIDI to LilyPond conversion:", e)
        sys.exit(1)

    # Optional: Modify the LilyPond file using mingus if you need to add headers or adjust notation.
    # For example, here we prepend a LilyPond version header.
    try:
        with open(ly_file, "r") as f:
            ly_content = f.read()

        # Create a header (mingus could generate more complex headers if desired)
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
        # new_ly_content = header + ly_content

        with open(ly_file, "w") as f:
            f.write(new_ly_content)
    except Exception as e:
        print("Error modifying the LilyPond file:", e)
        sys.exit(1)

    try:
        # Step 2: Use LilyPond to generate a PNG image from the LilyPond file.
        print("Generating PNG from LilyPond file...")
        subprocess.run(["lilypond", "-fpng", "-o", folder, ly_file], check=True)
    except subprocess.CalledProcessError as e:
        print("Error during PNG generation:", e)
        sys.exit(1)

    # Note: LilyPond typically names the output file based on the .ly filename.
    # Check for a PNG file with the same base name.
    png_file = f"{folder}/{output_prefix}.png"
    if os.path.exists(png_file):
        print(f"Success! PNG file created: {png_file}")
    else:
        print("PNG file was not created as expected.")


# Example usage:
if __name__ == "__main__":
    # Replace 'example.mid' with the path to your MIDI file.
    midi_path = "music/"
    midi_file_name = "traditional-scarborough-fair.midi"
    full_path = midi_path + midi_file_name
    name, _ = os.path.splitext(full_path)
    try:
        os.mkdir(name)
    except FileExistsError:
        pass

    convert_midi_to_png(full_path, name, output_prefix="sheet_music")
