from fontTools.varLib.instancer import __main__
from ultralytics import YOLO
from PIL import Image

from src.music_program.utils.global_variables import *

model = YOLO("runs/detect/train-13/weights/best.pt")
path_to_image = "src/all_data/generated/my_complex_images_test/my_midi_images/my_midi_files/song_513/song_513-1.png"

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

            y_1 = max(0, position[1] - margin)
            y_2 = min(image.height, position[3] + margin)

            crop_box = (0, y_1, image.width, y_2)
            cropped_image = image.crop(crop_box)

            canvas = Image.new("RGB", (image.width, int(image.height/3)), (255, 255, 255))

            paste_position_y = int(image.height/6) - int((y_2 - y_1)/2)
            paste_position = (0, paste_position_y)
            canvas.paste(cropped_image, paste_position)

            canvas = canvas.resize((WIDTH, int(HEIGHT/3)))

            staff_list.append((position[1], canvas))
            canvas.show()

        staff_list.sort(key=lambda x: x[0])

    return staff_list

if __name__ == "__main__":
    segment_image(path_to_image)
