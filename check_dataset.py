import os


def check_dataset(train_dir, val_dir):
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
               "A", "B", "C", "D", "E", "F", "G", "H", "K", "L",
               "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"]
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")

    for split, split_dir in [("train", train_dir), ("val", val_dir)]:
        print(f"\nChecking {split} dataset:")
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.exists(cls_dir):
                num_images = len([f for f in os.listdir(cls_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                print(f"Class {cls}: {num_images} images")
            else:
                print(f"Class {cls}: Missing")

    return classes


# Chạy kiểm tra
train_dir = r"D:\Bien_So\char_data\train"
val_dir = r"D:\Bien_So\char_data\val"
classes = check_dataset(train_dir, val_dir)