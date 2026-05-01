# test_kfs.py
from ultralytics import YOLO
import cv2
import torch
import sys
import os
import urllib.request

MODEL_PATH  = "runs/detect/kfs3/weights/best.pt"
CONF_THRESH = 0.5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

def download_from_url(url):
    """Download image or video from URL to a temp file."""
    ext = url.split("?")[0].split(".")[-1].lower()
    if ext not in ["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]:
        ext = "jpg"  # default to jpg if can't detect
    temp_path = f"temp_download.{ext}"
    print(f"Downloading from URL...")
    urllib.request.urlretrieve(url, temp_path)
    print(f"Downloaded to {temp_path}")
    return temp_path

def test_image(image_path):
    model = YOLO(MODEL_PATH)
    results = model(image_path, conf=CONF_THRESH, device=DEVICE)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        conf = float(box.conf[0])
        label = results[0].names[int(box.cls[0])]
        print(f"{label} at ({cx}, {cy}) | conf: {conf:.2f}")

    results[0].save(filename="result_image.jpg")
    print("Saved to result_image.jpg")


def test_video(video_path):
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)

    cap = cv2.VideoCapture(video_path)

    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        "result_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    print(f"Processing video: {total} frames at {fps} FPS")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRESH, verbose=False, device=DEVICE)
        annotated = results[0].plot()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            conf = float(box.conf[0])
            label = results[0].names[int(box.cls[0])]
            print(f"Frame {frame_count} | {label} at ({cx}, {cy}) | conf: {conf:.2f}")

        out.write(annotated)
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{total} frames...")

    cap.release()
    out.release()
    print(f"Done! Saved to result_video.mp4")


def is_url(path):
    return path.startswith("http://") or path.startswith("https://")

def is_video(path):
    return path.split("?")[0].split(".")[-1].lower() in ["mp4", "avi", "mov", "mkv"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_kfs.py path/to/image.jpg")
        print("  python test_kfs.py path/to/video.mp4")
        print("  python test_kfs.py https://example.com/image.jpg")
        print("  python test_kfs.py https://example.com/video.mp4")
        sys.exit()

    path = sys.argv[1]
    temp_file = None

    print(f"Using device: {DEVICE}")

    # Download if URL
    if is_url(path):
        temp_file = download_from_url(path)
        path = temp_file

    elif not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit()

    # Auto detect image or video
    if is_video(path):
        test_video(path)
    else:
        test_image(path)

    # Cleanup temp file
    if temp_file and os.path.exists(temp_file):
        os.remove(temp_file)