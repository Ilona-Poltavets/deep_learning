import os
import shutil

# CLASS_NAMES = ['Cat', 'Dog', 'Cow', 'Goat', 'Hen', 'Rabbit']
CLASS_NAMES = ['Cat', 'Dog', 'Hen', 'Rabbit']
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

def prepare_yolo_dataset(root_dir, output_dir):
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

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

        print(images_src)
        for image_file in os.listdir(images_src):
            shutil.copy(os.path.join(images_src, image_file), images_dir)

        for label_file in os.listdir(labels_src):
            label_path = os.path.join(labels_src, label_file)
            new_label_path = os.path.join(labels_dir, label_file)

            with open(label_path, 'r') as f_in, open(new_label_path, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x, y, w, h = parts
                        f_out.write(f"{class_id} {x} {y} {w} {h}\n")

    print(f"✅ YOLO dataset prepared at: {output_dir}")

original_dataset_path = '../datasets/animals/train'
yolo_output_path = '../datasets/animals/yolo_format'

prepare_yolo_dataset(original_dataset_path, yolo_output_path)
