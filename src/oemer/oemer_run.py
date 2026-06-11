import subprocess
import os

import onnxruntime as ort

import time
import re


def run_oemer(image_path, output_folder):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    command = ["oemer", image_path, "-o", output_folder]

    print(f"Running oemer with command: {command}")

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Success! Finished processing {image_path}.")
    else:
        print(f"Error processing {image_path}:")
        print(result.stderr)


if __name__ == "__main__":
    image_folder_path = "src/all_data/data_to_analyze/low_res"
    output_folder_path = "src/all_data/model_generated/oemer"
    previous_number = 42
    skipped_numbers = [21, 22, 32, 33, 38, 40]

    for folder in os.listdir(image_folder_path):
        number = int(re.findall(r'\d+', folder)[0])
        if (previous_number < number < 100) or (number in skipped_numbers):
            print(number)
            subfolder_path = os.path.join(image_folder_path, folder)
            for file in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, file)
                output_path = os.path.join(output_folder_path, folder)

                start = time.time()
                run_oemer(image_path, output_path)
                end = time.time()

                time_taken = end - start
                print(f"Time taken: {time_taken}s\n")



