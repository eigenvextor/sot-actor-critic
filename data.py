import os
import glob
import random
import cv2
from torch.utils.data import Dataset, DataLoader

class OTBSequenceDataset(Dataset):
    def __init__(self, sequence_dirs):
        self.sequence_dirs = sequence_dirs

    def get_groundtruth(self, filepath):
        """
        clean the ground truth data
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        boxes = []
        for line in lines:
            # replace tabs with spaces and split
            parts = line.strip().replace('\t', ',').split(',')
            if len(parts) < 4:
                # if space separated
                parts = line.strip().split()
            
            # otb dataset: [top_left_x, top_left_y, w, h]
            tl_x, tl_y, w, h = [float(p) for p in parts[:4]]
            
            cx = tl_x + (w / 2.0)
            cy = tl_y + (h / 2.0)
            
            # [cx, cy, w, h]
            boxes.append([cx, cy, w, h])

        return boxes

    def __len__(self):
        return len(self.sequence_dirs)

    def __getitem__(self, idx):
        seq_dir = self.sequence_dirs[idx]
        
        # load image paths and ground truth
        img_dir = os.path.join(seq_dir, 'img')
        img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        gt_path = os.path.join(seq_dir, 'groundtruth_rect.txt')
        
        ground_truths = self.get_groundtruth(gt_path)
        
        # handling frames/ ground truth no mismatch
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
            
        return frames, ground_truths

def get_otb_dataloaders(otb_root_dir, split_ratio=0.99):
    """
    scans the OTB root directory, splits the videos, and returns DataLoaders
    """
    # find all sequence directories
    all_seqs = []
    exclude_list = ['Skating2', 'Panda', 'Jogging', 'Human4']
    for d in os.listdir(otb_root_dir):
        if d.startswith('.'):
            continue
        dir_path = os.path.join(otb_root_dir, d)
        if os.path.isdir(dir_path) and dir_path.split('/')[-1] not in exclude_list:
            if os.path.exists(os.path.join(dir_path, 'groundtruth_rect.txt')):
                all_seqs.append(dir_path)

    # shuffle for a random split
    random.shuffle(all_seqs)
    
    split_idx = int(len(all_seqs) * split_ratio)
    train_seqs = all_seqs[:split_idx]
    test_seqs = all_seqs[split_idx:]
    
    print(f"Total Sequences: {len(all_seqs)}")
    print(f"Training on: {len(train_seqs)} sequences")
    print(f"Testing on: {len(test_seqs)} sequences")
    
    # create datasets
    train_dataset = OTBSequenceDataset(train_seqs)
    test_dataset = OTBSequenceDataset(test_seqs)
    
    # Create DataLoaders
    # Note: A custom collate_fn is usually needed if batch_size > 1 because 
    # images might have different native resolutions across different OTB videos.
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)
    
    return train_loader, test_loader