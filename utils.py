import cv2
import numpy as np
import torch
import torchvision.transforms as T
from collections import deque
import torch.nn.functional as F

# convert to pytorch tensor and normalize using ImageNet statistics
PATCH_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_state_patch(frame, bbox, target_size=107, device='cpu'):
    if not isinstance(bbox, torch.Tensor):
        bbox = torch.tensor(bbox, dtype=torch.float32, device=device)

    h_frame, w_frame = frame.shape[:2]
    x_c, y_c, w, h = bbox
    
    # absolute pixel coords for the crop
    x1 = int(torch.round(x_c - w/2))
    y1 = int(torch.round(y_c - h/2))
    x2 = int(torch.round(x_c + w/2))
    y2 = int(torch.round(y_c + h/2))

    # oob padding
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w_frame)
    pad_bottom = max(0, y2 - h_frame)

    # valid coords
    valid_x1 = max(0, x1)
    valid_y1 = max(0, y1)
    valid_x2 = min(w_frame, x2)
    valid_y2 = min(h_frame, y2)

    # crop the valid region from the frame
    valid_crop = frame[valid_y1:valid_y2, valid_x1:valid_x2]

    if valid_crop.size == 0:
        patch = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    else:
        # apply padding if the box went outside the frame boundaries
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            # pad with the mean RGB value (approx 128)
            patch = cv2.copyMakeBorder(
                valid_crop, 
                pad_top, pad_bottom, pad_left, pad_right, 
                cv2.BORDER_CONSTANT, 
                value=(128, 128, 128)
            )
            # print("padding done")
        else:
            patch = valid_crop

    # edge case where box dimensions are 0 or negative
    if patch is None or patch.size == 0:
        # print("patch is none")
        patch = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # resize to the network's required input size (107x107)
    patch_resized = cv2.resize(patch, (target_size, target_size))
    
    # add batch dimension: shape becomes [1, 3, 107, 107]
    state_tensor = PATCH_TRANSFORM(patch_resized).unsqueeze(0).to(device)
    
    return state_tensor

def apply_action(bbox, action_tensor):
    if isinstance(action_tensor, torch.Tensor):
        # dx, dy, dw, dh = action_tensor.squeeze().cpu().detach().numpy().tolist()
        dx, dy, ds = action_tensor.squeeze().cpu().detach().numpy().tolist()
    else:
        # dx, dy, dw, dh = action_tensor
        dx, dy, ds = action_tensor

    x, y, w, h = bbox
    
    # updates
    new_x = x + dx * w
    new_y = y + dy * h
    new_w = w + ds * w
    new_h = h + ds * h
    
    return [new_x, new_y, new_w, new_h]

def apply_action_test(bbox, action_tensor, first_box):    
    if isinstance(action_tensor, torch.Tensor):
        # dx, dy, dw, dh = action_tensor.squeeze().cpu().detach().numpy().tolist()
        dx, dy, ds = action_tensor.squeeze().cpu().detach().numpy().tolist()
    else:
        # dx, dy, dw, dh = action_tensor
        dx, dy, ds = action_tensor

    x, y, w, h = bbox
    _, _, w0, h0 = first_box

    # updates
    new_x = x + dx * w
    new_y = y + dy * h
    # new_w = max(0.99*w0, min(1.01*w0, w * (1 + dw)))
    # new_h = max(0.99*h0, min(1.01*h0, h * (1 + dh)))
    new_w = max(0.99*w0, min(1.01*w0, w * (1 + ds)))
    new_h = max(0.99*h0, min(1.01*h0, h * (1 + ds))) 
    
    return [new_x, new_y, new_w, new_h]

def get_expert_action(current_bbox, gt_bbox, device, clipped=False):
    x_c, y_c, w_c, h_c = current_bbox
    x_gt, y_gt, w_gt, h_gt = gt_bbox
    
    wh_c = (w_c + h_c)/2

    # eq1
    dx = (x_gt - x_c) / w_c
    dy = (y_gt - y_c) / h_c
    # dw = (w_gt - w_c) / w_c
    # dh = (h_gt - h_c) / h_c
    ds = (w_gt - w_c) / wh_c
    
    # ground truth action shouldn't be clipped. its ground truth, duh
    if clipped:
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))
        # dw = max(-0.05, min(0.05, dw))
        # dh = max(-0.05, min(0.05, dh))
        ds = max(-0.05, min(0.05, ds))

    # action = torch.tensor([[dx, dy, dw, dh]], dtype=torch.float32, device=device)
    action = torch.tensor([[dx, dy, ds]], dtype=torch.float32, device=device)
    
    return action

def generate_actor_samples(gt_bbox, num_samples=64, iou_threshold=0.7):
    samples = []
    x, y, w, h = gt_bbox
    
    x_std = 0.3
    y_std = 0.3
    scale_std = 0.5
    
    while len(samples) < num_samples:
        # generate samples
        dx = np.random.normal(0, x_std)
        dy = np.random.normal(0, y_std)
        # dw = np.random.normal(0, scale_std)
        # dh = np.random.normal(0, scale_std)
        ds = np.random.normal(0, scale_std)
        
        # apply the translations directly, and the scale as a relative multiplier
        noisy_bbox = [x + dx * w, y + dy * h, w * (1 + ds), h * (1 + ds)]
        
        # enforce the strict IoU threshold from the paper
        if calculate_iou(noisy_bbox, gt_bbox) > iou_threshold:
            samples.append(noisy_bbox)
            
    return samples

def generate_critic_samples(gt_bbox, num_pos_samples=50, num_neg_samples=100, iou_threshold=0.7):
    pos_samples, neg_samples = [], []
    x, y, w, h = gt_bbox

    pos_x_std, neg_x_std = 0.3, 0.3
    pos_y_std, neg_y_std = 0.3, 0.3
    pos_scale_std, neg_scale_std = 0.5, 0.5

    while len(pos_samples) < num_pos_samples:
        # generate samples
        dx = np.random.normal(0, pos_x_std)
        dy = np.random.normal(0, pos_y_std)
        # dw = np.random.normal(0, pos_scale_std)
        # dh = np.random.normal(0, pos_scale_std)
        ds = np.random.normal(0, pos_scale_std)
        
        # apply the translations directly, and the scale as a relative multiplier
        noisy_bbox = [x + dx * w, y + dy * h, w * (1 + ds), h * (1 + ds)]
        
        # enforce the strict IoU threshold from the paper
        if calculate_iou(noisy_bbox, gt_bbox) > iou_threshold:
            pos_samples.append(noisy_bbox)

    while len(neg_samples) < num_neg_samples:
        # generate samples
        dx = np.random.normal(0, neg_x_std)
        dy = np.random.normal(0, neg_y_std)
        # dw = np.random.normal(0, neg_scale_std)
        # dh = np.random.normal(0, neg_scale_std)
        ds = np.random.normal(0, neg_scale_std)
        
        # apply the translations directly, and the scale as a relative multiplier
        noisy_bbox = [x + dx * w, y + dy * h, w * (1 + ds), h * (1 + ds)]
        
        # enforce the strict IoU threshold from the paper
        if calculate_iou(noisy_bbox, gt_bbox) < 1 - iou_threshold:
            neg_samples.append(noisy_bbox)
        
    return pos_samples, neg_samples

def calculate_iou(box_a, box_b):
    # convert [x_center, y_center, w, h] to [x_min, y_min, x_max, y_max]
    # (bottom left, top right)
    def get_corners(b):
        return [b[0] - b[2]/2, b[1] - b[3]/2, b[0] + b[2]/2, b[1] + b[3]/2]
    
    rect_a = get_corners(box_a)
    rect_b = get_corners(box_b)
    
    xl = max(rect_a[0], rect_b[0])
    yl = max(rect_a[1], rect_b[1])
    xr = min(rect_a[2], rect_b[2])
    yr = min(rect_a[3], rect_b[3])
        
    interArea = max(0, xr - xl) * max(0, yr - yl)
    
    box_a_area = box_a[2] * box_a[3]
    box_b_area = box_b[2] * box_b[3]
    
    iou = interArea / float(box_a_area + box_b_area - interArea + 1e-6)
    return iou
