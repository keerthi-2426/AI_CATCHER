import os
import cv2
import random
import shutil
from pathlib import Path

# ===== EDIT THESE PATHS =====
VIDEOS_ROOT = r"C:\Users\kugan\Downloads\fmodel\Real Life Violence Dataset"  # <-- your folder
OUT_ROOT = r"C:\Users\kugan\fight_dataset_cls"  # output images dataset

# ===== SETTINGS =====
CLASSES = {
    "Violence": "violence",
    "NonViolence": "nonviolence",
}

TRAIN_RATIO = 0.85
FRAMES_PER_VIDEO = 12       # extract 12 frames per video (increase for better accuracy)
MIN_GAP_SEC = 0.3           # minimum time gap between sampled frames
MAX_VIDEOS_PER_CLASS = 0    # 0 = use all videos, else limit (debug)

SEED = 42


def list_videos(folder: str):
    exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
    paths = []
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(str(p))
    return sorted(paths)


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def extract_frames(video_path: str, out_dir: str, frames_per_video: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ cannot open:", video_path)
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if frame_count > 0 else 0

    # pick timestamps uniformly
    if duration <= 0.1:
        # fallback: sample by frame index
        indexes = set()
        step = max(1, frame_count // max(1, frames_per_video))
        for i in range(0, frame_count, step):
            indexes.add(i)
            if len(indexes) >= frames_per_video:
                break
        targets = sorted(indexes)
        mode = "index"
    else:
        # sample timestamps with minimum gap
        targets = []
        t = 0.0
        step = max(MIN_GAP_SEC, duration / max(1, frames_per_video))
        while t < duration and len(targets) < frames_per_video:
            targets.append(t)
            t += step
        mode = "time"

    saved = 0
    base = Path(video_path).stem

    if mode == "time":
        for j, ts in enumerate(targets):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            out_path = os.path.join(out_dir, f"{base}_{j:03d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
    else:
        for j, idx in enumerate(targets):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            out_path = os.path.join(out_dir, f"{base}_{j:03d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1

    cap.release()
    return saved


def main():
    random.seed(SEED)

    # Create output structure
    for split in ["train", "val"]:
        for cname in CLASSES.values():
            safe_mkdir(os.path.join(OUT_ROOT, split, cname))

    for src_class_folder, dst_class_name in CLASSES.items():
        src_dir = os.path.join(VIDEOS_ROOT, src_class_folder)
        videos = list_videos(src_dir)

        if MAX_VIDEOS_PER_CLASS and len(videos) > MAX_VIDEOS_PER_CLASS:
            videos = random.sample(videos, MAX_VIDEOS_PER_CLASS)

        random.shuffle(videos)
        n_train = int(len(videos) * TRAIN_RATIO)
        train_v = videos[:n_train]
        val_v = videos[n_train:]

        print(f"\nClass {dst_class_name}: {len(videos)} videos | train {len(train_v)} | val {len(val_v)}")

        for split_name, split_list in [("train", train_v), ("val", val_v)]:
            out_dir = os.path.join(OUT_ROOT, split_name, dst_class_name)
            for vp in split_list:
                saved = extract_frames(vp, out_dir, FRAMES_PER_VIDEO)
                if saved == 0:
                    print("⚠️ no frames:", vp)

    print("\n✅ Done! Dataset created at:", OUT_ROOT)
    print("Expected structure:")
    print(rf"{OUT_ROOT}\train\violence\*.jpg")
    print(rf"{OUT_ROOT}\train\nonviolence\*.jpg")
    print(rf"{OUT_ROOT}\val\violence\*.jpg")
    print(rf"{OUT_ROOT}\val\nonviolence\*.jpg")


if __name__ == "__main__":
    main()