from ultralytics import YOLO

model = YOLO("runs/detect/train-9/weights/best.pt")

path_to_image = "src/all_data/generated/my_complex_images_test/my_midi_images/my_midi_files/song_1/song_1-1.png"

results = model(path_to_image, imgsz=640)

for result in results:
    boxes = result.boxes

    for box in boxes:
        position = box.xyxy[0].tolist()
        position = [round(x) for x in position]

        confidence = round(float(box.conf[0]), 2)
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        print(f"Znaleziono obiekt: '{class_name}'")
        print(f" -> Pozycja na obrazie (piksele): {position}")
        print(f" -> Pewność: {confidence}\n")

    result.save(filename="wynik_detekcji.jpg")