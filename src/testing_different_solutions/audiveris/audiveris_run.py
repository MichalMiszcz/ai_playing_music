### Obrazy dla Audiveris zostały przeskalowane do wyższej rozdzielczości, jednocześnie nie zwiększając informacji o obrazie
### Zostało to zastosowane w celu poprawnego działania Audiveris


import subprocess
import os

from PIL import Image
from music21 import converter, tempo, instrument

import time
import re


def run_audiveris(image_path, output_folder):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    command = ["audiveris", "-batch", "-export", "-option", "org.audiveris.omr.sheet.SheetStub.minInterline=0", "-output", output_folder, image_path]

    print(f"Running oemer with command: {command}")

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Success! Finished processing {image_path}.")
    else:
        print(f"Error processing {image_path}:")
        print(result.stderr)


if __name__ == "__main__":
    image_folder_path = "src/all_data/data_to_analyze/low_res"
    new_image_folder_path = "src/all_data/data_to_analyze/hi_scaled_res"
    output_folder_path = "src/all_data/model_generated/audiveris"
    previous_number = 0
    max_num = 100
    skipped_numbers = []

    for folder in os.listdir(image_folder_path):
        number = int(re.findall(r'\d+', folder)[0])
        if (previous_number < number <= max_num) or (number in skipped_numbers):
            print(number)
            subfolder_path = os.path.join(image_folder_path, folder)
            for file in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, file)
                new_image_path = os.path.join(new_image_folder_path, file)
                output_path = os.path.join(output_folder_path, folder)

                start = time.time()
                with Image.open(image_path) as img:
                    if not os.path.exists(new_image_path):
                        print("Resizing image")
                        new_size = (img.width * 4, img.height * 4)
                        img = img.resize(new_size, Image.Resampling.NEAREST)
                        img.save(new_image_path)

                image_path = new_image_path

                if not os.path.exists(output_path):
                    run_audiveris(image_path, output_path)
                end = time.time()
                time_taken = end - start
                print(f"Time taken: {time_taken}s\n")

                output_midi_folder = os.path.join(output_folder_path, 'midi')
                if not os.path.exists(output_midi_folder):
                    os.makedirs(output_midi_folder)

                mxml_name = folder + "-1.mxl"
                input_mxml = os.path.join(output_path, mxml_name)
                score = converter.parse(input_mxml)
                score.insert(0, tempo.MetronomeMark(number=120))

                for part in score.parts:
                    part.insert(0, instrument.Piano())

                midi_name = folder + ".midi"
                output_midi = os.path.join(output_midi_folder, midi_name)
                if not os.path.exists(output_midi):
                    score.write("midi", output_midi)



