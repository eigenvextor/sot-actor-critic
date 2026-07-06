import os
import sys
import glob
import cv2
import torch
from track import (
    get_groundtruth,
    load_checkpoint,
    online_tracking
)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"executing on device: {device}")

    OTB_ROOT_DIR = "OTB2015"
    CHECKPOINT_DIR = input("which directory: ")  
    CHECKPOINT = input("which checkpoint: ")

    print(f"loading ACT model: {CHECKPOINT}")
    model = load_checkpoint(CHECKPOINT_DIR, CHECKPOINT, device)

    if not model:
        sys.exit(0)

    video_seq = input("give input video seq name: ")

    video_seq_path = os.path.join(OTB_ROOT_DIR, video_seq)
    if os.path.exists(video_seq_path):
        img_dir = os.path.join(video_seq_path, 'img')
        img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        gt_path = os.path.join(video_seq_path, 'groundtruth_rect.txt')

        ground_truths = get_groundtruth(gt_path)

        # handling frames/ ground truth mismatch
        total_frames = len(img_paths)
        valid_frames = min(total_frames, len(ground_truths))

        img_paths = img_paths[:valid_frames]
        ground_truths = ground_truths[:valid_frames]

        # load actual images into memory
        frames = []
        for path in img_paths:
            # cv reads in gbr format but rgb needed
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        online_tracking(model, frames, ground_truths, device=device)

    else:
        print(f"video seq {video_seq} doesnt exist!")
        
    