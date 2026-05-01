import os
import shutil
import random
from ultralytics import YOLO
import torch

def rearrange_split(dataset_path="KFS", train_ratio=0.85, val_ratio=0.10):
    print("Rearranging dataset split...")
    
    all_images = []
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset_path, split, "images")
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                if f.endswith((".jpg", ".jpeg", ".png")):
                    all_images.append(f)

    random.seed(42)
    random.shuffle(all_images)
    total = len(all_images)
    train_end = int(total * train_ratio)
    val_end   = int(total * (train_ratio + val_ratio))

    splits = {
        "train": all_images[:train_end],
        "valid": all_images[train_end:val_end],
        "test":  all_images[val_end:]
    }

    print(f"Total: {total} | Train: {len(splits['train'])} | Valid: {len(splits['valid'])} | Test: {len(splits['test'])}")

    def find_file(filename, folder):
        for split in ["train", "valid", "test"]:
            path = os.path.join(dataset_path, split, folder, filename)
            if os.path.exists(path):
                return path
        return None

    temp_path = dataset_path + "_temp"
    for split_name, files in splits.items():
        os.makedirs(os.path.join(temp_path, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(temp_path, split_name, "labels"), exist_ok=True)
        for img_file in files:
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_img = find_file(img_file, "images")
            src_lbl = find_file(label_file, "labels")
            if src_img:
                shutil.copy2(src_img, os.path.join(temp_path, split_name, "images", img_file))
            if src_lbl:
                shutil.copy2(src_lbl, os.path.join(temp_path, split_name, "labels", label_file))

    for split in ["train", "valid", "test"]:
        shutil.rmtree(os.path.join(dataset_path, split), ignore_errors=True)
        shutil.copytree(os.path.join(temp_path, split), os.path.join(dataset_path, split))
    shutil.rmtree(temp_path)

    print("Split rearranged successfully!")


def train():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {DEVICE}")

    # ── Resume from existing weights ─────────────────────────────────────────
    resume_weights = "runs/detect/kfs3/weights/last.pt"  # last.pt to resume, best.pt to finetune
    
    if os.path.exists(resume_weights):
        print(f"Resuming from {resume_weights}")
        model = YOLO(resume_weights)
        model.train(
            data="KFS/data.yaml",
            epochs=30,
            imgsz=640,
            batch=16,
            device=DEVICE,
            patience=10,
            save=True,
            plots=True,
            name="kfs",
            workers=0,
            resume=True,       # picks up exactly where it left off
        )
    else:
        print(f"No weights found at {resume_weights}, starting fresh...")
        model = YOLO("yolov8n.pt")
        model.train(
            data="KFS/data.yaml",
            epochs=30,
            imgsz=640,
            batch=16,
            device=DEVICE,
            patience=10,
            save=True,
            plots=True,
            name="kfs",
            workers=0,
        )

    print("Done! Weights at runs/detect/kfs/weights/best.pt")


if __name__ == "__main__":
    rearrange_split()
    train()