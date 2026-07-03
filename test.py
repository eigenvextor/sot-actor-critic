import os
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
    CHECKPOINT = "500000"

    print(f"loading ACT model: {CHECKPOINT}")
    model = load_checkpoint(CHECKPOINT, device)

    # # done only once to save video_seq.txt file
    # all_seqs = []
    # exclude_list = ['Skating2', 'Panda', 'Jogging', 'Human4']
    # for d in sorted(os.listdir(OTB_ROOT_DIR)):
    #     if d.startswith('.'):
    #         continue
    #     dir_path = os.path.join(OTB_ROOT_DIR, d)
    #     if os.path.isdir(dir_path) and dir_path.split('/')[-1] not in exclude_list:
    #         if os.path.exists(os.path.join(dir_path, 'groundtruth_rect.txt')):
    #             all_seqs.append(d)

    # with open('videos_seq.txt', 'w') as f:
    #     for i, video in enumerate(all_seqs):
    #         f.write(f"{i}, {video}\n")

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

        # only pass the first frame ground truth
        online_tracking(model, frames, ground_truths[0], device=device)

    else:
        print(f"video seq {video_seq} doesnt exist!")
        
    