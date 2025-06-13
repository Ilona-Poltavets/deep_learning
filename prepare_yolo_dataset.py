import os
import shutil


def prepare_yolo_dataset(root_dir, output_dir):
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        class_path=os.path.join(class_path, 'train')
        if os.path.isdir(class_path):
            images_src = os.path.join(class_path, 'images')
            labels_src = os.path.join(class_path, 'labels')

            if os.path.exists(images_src):
                for image_file in os.listdir(images_src):
                    shutil.copy(os.path.join(images_src, image_file), images_dir)

            if os.path.exists(labels_src):
                for label_file in os.listdir(labels_src):
                    shutil.copy(os.path.join(labels_src, label_file), labels_dir)

    print(f"YOLO dataset prepared at: {output_dir}")


original_dataset_path = '../datasets/animals/train'
yolo_output_path = '../datasets/animals/yolo_format'

prepare_yolo_dataset(original_dataset_path, yolo_output_path)
