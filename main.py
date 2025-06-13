from ultralytics import YOLO
import torch
import os
import yaml

# print(torch.cuda.get_device_name(0))
# print(torch.cuda.device_count())
# print(torch.cuda.is_available())
# print(torch.version.cuda)

def check_dataset(data_path):
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)

    train_path = data_config.get('train')
    val_path = data_config.get('val')

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise FileNotFoundError(f"Train or validation path does not exist in {data_path}")

    def count_images(path):
        return sum(1 for file in os.listdir(path) if file.endswith(('.jpg', '.jpeg', '.png')))

    if count_images(train_path) == 0 or count_images(val_path) == 0:
        raise ValueError("No images found in train or val dataset directories.")


if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    data_path = './data.yaml'

    try:
        check_dataset(data_path)
        print("Dataset validation passed.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Dataset check failed: {e}")
        exit(1)

    model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name='animal_classification',
        workers=4,
        device=0
    )

    test_image = '../datasets/test_image.jpg'
    results = model.predict(source=test_image, show=True)

