from ultralytics import YOLO
from PIL import Image

from src.music_program.utils.global_variables import *

model = YOLO("runs/detect/train-13/weights/best.pt")
path_to_image = "src/all_data/generated/my_complex_images_test/my_midi_images/my_midi_files/song_513/song_513-1.png"
# path_to_image = "src/all_data/data_to_analyze/hi_scaled_res/song_1-1.png"

CANVAS_HEIGHT = 130
CANVAS_WIDTH = 585

def segment_image(path_to_image=None, source_image=None):
    if path_to_image is None and source_image is not None:
        image = source_image
    elif path_to_image is not None:
        image = Image.open(path_to_image)
    else:
        raise Exception

    results = model(image, imgsz=640, conf=0.5)

    staff_list = []
    for result in results:
        boxes = result.boxes

        for box in boxes:
            position = box.xyxy[0].tolist()
            position = [round(x) for x in position]

            margin = int((position[3] - position[1])/4)

            img_width, img_height = image.size
            y_1 = max(0, position[1] - margin)
            y_2 = min(img_height, position[3] + margin)

            crop_box = (0, y_1, img_width, y_2)
            cropped_image = image.crop(crop_box)

            scale = (CANVAS_WIDTH/img_width)
            cropped_new_y = int(scale * cropped_image.height)

            # Wycinek obrazka jest skalowany tak, aby szerokością pasował do standardowej szerokości wycinka
            cropped_image = cropped_image.resize((CANVAS_WIDTH, cropped_new_y))

            # canvas musi mieć stałe proporcje (bardzo zbliżone do proporcji obrazów w zbiorze uczącym),
            # ponieważ chcę go potem przeskalować w specyficzny sposób, tak jak były skalowane obrazy podczas uczenia
            canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (255, 255, 255))

            paste_position_y = int(CANVAS_HEIGHT/2) - int(cropped_new_y/2)
            paste_position = (0, paste_position_y)
            canvas.paste(cropped_image, paste_position)
            canvas = canvas.resize((WIDTH, int(HEIGHT/3)))
            # canvas.show()

            staff_list.append((position[1], canvas))

        staff_list.sort(key=lambda x: x[0])

    return staff_list

if __name__ == "__main__":
    segment_image(path_to_image)
