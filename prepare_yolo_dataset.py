import os
import shutil

CLASS_NAMES = ['Cat', 'Dog', 'Hen', 'Rabbit']
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

def prepare_yolo_dataset(root_dir, output_dir, output_dir_val):
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')

    images_dir_val = os.path.join(output_dir_val, 'images')
    labels_dir_val = os.path.join(output_dir_val, 'labels')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir_val, exist_ok=True)
    os.makedirs(labels_dir_val, exist_ok=True)

    for class_name in os.listdir(root_dir):
        class_folder = os.path.join(root_dir, class_name, 'train')

        print(f"Preparing {class_name}...")
        if not os.path.isdir(class_folder):
            continue

        class_id = CLASS_NAME_TO_ID.get(class_name)
        if class_id is None:
            continue

        images_src = os.path.join(class_folder, 'images')
        labels_src = os.path.join(class_folder, 'labels')

        image_files = sorted(os.listdir(images_src))
        counter = 0

        for image_file in image_files:
            image_path = os.path.join(images_src, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_src, label_file)

            if counter % 5 == 0:
                dst_images = images_dir_val
                dst_labels = labels_dir_val
            else:
                dst_images = images_dir
                dst_labels = labels_dir

            shutil.copy(image_path, os.path.join(dst_images, image_file))

            if os.path.exists(label_path):
                with open(label_path, 'r') as f_in, open(os.path.join(dst_labels, label_file), 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, x, y, w, h = parts
                            f_out.write(f"{class_id} {x} {y} {w} {h}\n")

            counter += 1

    print(f"✅ YOLO dataset prepared:\n - train: {output_dir}\n - val: {output_dir_val}")

original_dataset_path = '../datasets/animals/train'
yolo_output_path = '../datasets/animals/yolo_format'
yolo_output_path_val = '../datasets/animals/yolo_format_val'

prepare_yolo_dataset(original_dataset_path, yolo_output_path, yolo_output_path_val)
