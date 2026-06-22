from ultralytics import YOLO

path_to_images = "src/all_data/labeled_images/data.yaml"
yolo_version = "yolo26x.pt"

epochs = 25
patience = 0
batch_size = 8
workers = 4

def main():
    model = YOLO(yolo_version)
    model.train(
        data=path_to_images, epochs=epochs, patience=patience, imgsz=640, batch=batch_size, workers=workers
    )


if __name__ == "__main__":
    main()
