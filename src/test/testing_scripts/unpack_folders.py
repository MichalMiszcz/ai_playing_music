import os
import shutil

folder_to_change = 'src/all_data/data_to_analyze/low_res'

for root, dirs, files in os.walk(folder_to_change):
    for folder in dirs:
        dir_path = os.path.join(root, folder)
        number = int(''.join(filter(str.isdigit, dir_path)))
        print(number)

        for image in os.listdir(dir_path):
            image_path = os.path.join(dir_path, image)
            copy_image_folder = f'src/all_data/data_to_analyze/low_res_songs'
            copy_image_path = os.path.join(copy_image_folder, image)

            shutil.copy(image_path, copy_image_path)