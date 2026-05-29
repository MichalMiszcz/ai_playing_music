from ultralytics import YOLO
from PIL import Image

model = YOLO("runs/detect/train-13/weights/best.pt")

path_to_image = "src/all_data/generated/my_complex_images_test/my_midi_images/my_midi_files/song_513/song_513-1.png"

results = model(path_to_image, imgsz=640)

for result in results:
    boxes = result.boxes

    for box in boxes:
        position = box.xyxy[0].tolist()
        position = [round(x) for x in position]

        confidence = round(float(box.conf[0]), 2)
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        y_1 = position[1] - 8
        y_2 = position[3] + 8

        image = Image.open(path_to_image)
        crop_box = (0, y_1, image.width, y_2)
        cropped_image = image.crop(crop_box)

        canvas = Image.new("RGB", (image.width, int(image.height/3)), (255, 255, 255))

        paste_position_y = int(image.height/6) - int((y_2 - y_1)/2)
        paste_position = (0, paste_position_y)
        canvas.paste(cropped_image, paste_position)

        canvas = canvas.resize((512, int(512/3)))
