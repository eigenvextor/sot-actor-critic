import numpy as np
from collections import deque
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import utils
from model import ActorCriticTracker

def get_groundtruth(filepath):
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

def load_checkpoint(model_checkpoint, device):
    path = f"checkpoints/act_tracker_iter_{model_checkpoint}.pth"
    model = ActorCriticTracker()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model

def initialize_actor(model, initial_frame, initial_gt, num_samples=500, num_iterations=30, 
                     learning_rate=1e-4, iou_threshold=0.7, batch_size=64, device="cpu"):
    # 1e-4 LR specifically for the initialization stage
    actor_params = list(model.actor.parameters())
    init_optimizer = optim.Adam(actor_params, lr=learning_rate)
    
    mse_loss = nn.MSELoss()

    sampled_bboxes = utils.generate_actor_samples(initial_gt, num_samples=num_samples, iou_threshold=iou_threshold)
    
    state_tensors = []
    target_actions = []
    
    for bbox in sampled_bboxes:
        expert_action = utils.get_expert_action(bbox, initial_gt, device, clipped=True)
        state_tensor = utils.get_state_patch(initial_frame, bbox, device=device)
        
        state_tensors.append(state_tensor)
        target_actions.append(expert_action)

    # print(len(target_actions), target_actions[0])
        
    batch_states = torch.cat(state_tensors)
    batch_targets = torch.cat(target_actions)
    
    dataset = TensorDataset(batch_states, batch_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("actor initialization")
    
    for i in range(num_iterations):
        epoch_loss = 0.0
        for batch_s, batch_t in loader:
            with torch.no_grad():
                features = model.extract_actor_features(batch_s)

            predicted_actions = model.get_action(features)
            loss = mse_loss(predicted_actions, batch_t)
            
            init_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, max_norm=4.0)
            init_optimizer.step()

            epoch_loss += loss.item()

        if not (i+1) % 10:
            avg_loss = epoch_loss / len(loader)
            print(f"iteration {i+1}: {avg_loss:.6f}")

def initialize_critic(model, initial_frame, initial_gt, num_pos_samples=500, num_neg_samples=1000, 
                      num_iterations=30, learning_rate=1e-4, iou_threshold=0.7, batch_size=128, device="cpu"):
    # 1e-4 LR specifically for the initialization stage
    critic_params = list(model.c_fc1.parameters()) + \
                    list(model.c_fc2.parameters())
    init_optimizer = optim.Adam(critic_params, lr=learning_rate)
    
    bce_loss = nn.BCEWithLogitsLoss()

    pos_samples, neg_samples = utils.generate_critic_samples(initial_gt, num_pos_samples=num_pos_samples, 
                                                             num_neg_samples=num_neg_samples, iou_threshold=iou_threshold)

    batch_targets = torch.tensor([[1.0]] * num_pos_samples + [[0.0]] * num_neg_samples, dtype=torch.float32, device=device)
    sampled_bboxes = pos_samples + neg_samples

    state_tensors = []
    for bbox in sampled_bboxes:
        state_tensor = utils.get_state_patch(initial_frame, bbox, device=device)
        state_tensors.append(state_tensor)
    
    batch_states = torch.cat(state_tensors)

    dataset = TensorDataset(batch_states, batch_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("critic initialization")

    for i in range(num_iterations):
        epoch_loss = 0.0
        for batch_s, batch_t in loader:
            with torch.no_grad():
                features = model.extract_critic_features(batch_s)
        
            confidence_score = model.get_confidence_value_online(features)
            loss = bce_loss(confidence_score, batch_t)

            init_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_params, max_norm=1.0)
            init_optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        if avg_loss < 1e-4:
            print(f"iteration {i+1}: {avg_loss:.6f}")
            break
        
        if not (i+1) % 10:
            print(f"iteration {i+1}: {avg_loss:.6f}")

def redetection_strategy(model, frame, bbox, num_samples=256, device="cpu"):
    samples = []
    x, y, w, h = bbox
    
    x_std = 0.3
    y_std = 0.3
    scale_std = 0.5
    
    while len(samples) < num_samples:
        # generate samples
        dx = np.random.normal(0, x_std)
        dy = np.random.normal(0, y_std)
        dw = np.random.normal(0, scale_std)
        dh = np.random.normal(0, scale_std)
        
        # apply the translations directly, and the scale as a relative multiplier
        noisy_bbox = [x + dx * w, y + dy * h, w * (1 + dw), h * (1 + dh)]
        if noisy_bbox[2] > 0 and noisy_bbox[3] > 0:
            samples.append(noisy_bbox)

    state_tensors = []
    for cand_bbox in samples:
        state_tensor = utils.get_state_patch(frame, cand_bbox, device=device)
        state_tensors.append(state_tensor)

    batch_states = torch.cat(state_tensors)
    with torch.no_grad():
        batch_features = model.extract_critic_features(batch_states)

    confidence_scores = model.get_confidence_value_online(batch_features)
    idx = torch.argmax(confidence_scores)

    return confidence_scores[idx], samples[idx]

def online_critic_update(model, buffer, num_pos_samples=50, num_neg_samples=100, num_iterations=30, 
                         learning_rate=1e-5, iou_threshold=0.7, batch_size=128, device="cpu"):
    
    # freeze actor network; only critic networks are updated
    model.train()
    model.actor_feature_extractor.eval()
    model.critic_feature_extractor.eval()
    model.actor.eval()

    # 1e-5 LR specifically for the initialization stage
    critic_params = list(model.c_fc1.parameters()) + \
                    list(model.c_fc2.parameters())
    init_optimizer = optim.Adam(critic_params, lr=learning_rate)
    
    bce_loss = nn.BCEWithLogitsLoss()

    state_tensors, target_list = [], []
    for frame, ground_truth in buffer:
        pos_samples, neg_samples = utils.generate_critic_samples(ground_truth, num_pos_samples=num_pos_samples, 
                                                                num_neg_samples=num_neg_samples, iou_threshold=iou_threshold)

        frame_targets = torch.tensor([[1.0]] * num_pos_samples + [[0.0]] * num_neg_samples, dtype=torch.float32, device=device)
        target_list.append(frame_targets)
        sampled_bboxes = pos_samples + neg_samples

        for bbox in sampled_bboxes:
            state_tensor = utils.get_state_patch(frame, bbox, device=device)
            state_tensors.append(state_tensor)
        
    batch_states = torch.cat(state_tensors)
    batch_targets = torch.cat(target_list)

    dataset = TensorDataset(batch_states, batch_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("online critic update")

    for i in range(num_iterations):
        epoch_loss = 0.0
        for batch_s, batch_t in loader:
            with torch.no_grad():
                features = model.extract_critic_features(batch_s)
        
            confidence_score = model.get_confidence_value_online(features)
            loss = bce_loss(confidence_score, batch_t)

            init_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_params, max_norm=1.0)
            init_optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        if avg_loss < 1e-4:
            print(f"iteration {i+1}: {avg_loss:.6f}")
            break
        
        if not (i+1) % 10:
            print(f"iteration {i+1}: {avg_loss:.6f}")
    
    model.eval()

def online_tracking(model, frames, ground_truths, device): 
    model.train()
    model.actor_feature_extractor.eval()
    model.critic_feature_extractor.eval()
    
    # actor initialization
    initial_frame = frames[0]
    initial_gt = ground_truths[0]
    initialize_actor(model, initial_frame, initial_gt, num_samples=500, num_iterations=30, 
                     learning_rate=1e-4, iou_threshold=0.7, batch_size=64, device=device)
    
    # critic initialization
    initialize_critic(model, initial_frame, initial_gt, num_pos_samples=500, num_neg_samples=1000, 
                      num_iterations=30, learning_rate=1e-4, iou_threshold=0.7, batch_size=64, device=device)

    # subsequent frames are processed
    model.eval()

    buffer = deque(maxlen=10)

    current_bbox = initial_gt

    for current_frame, current_ground_truth in zip(frames[1:], ground_truths[1:]):
        
        # 1. obtain state s_t
        state_tensor = utils.get_state_patch(current_frame, current_bbox, device=device)

        # 2. select action a_t
        with torch.no_grad():
            state_features = model.extract_actor_features(state_tensor)
            action = model.get_action(state_features)

        # 3. execute action
        next_bbox = utils.apply_action(current_bbox, action)
        next_state_tensor = utils.get_state_patch(current_frame, next_bbox, device=device)

        # 4. get confidence value q-value
        with torch.no_grad():
            next_state_features = model.extract_critic_features(next_state_tensor)
            confidence_value = model.get_confidence_value_online(next_state_features)

        if confidence_value > 0:
            # go to next frame
            current_bbox = next_bbox
            buffer.append((current_frame, current_bbox))
        else:
            print(f"confidence value: {confidence_value}. applying redetection strategy")

            new_confidence_value, current_bbox = redetection_strategy(model, current_frame, current_bbox, num_samples=256, device=device)
 
            if new_confidence_value < 0:
                print(f"redetection strategy applied and still the new confidence score is only: {new_confidence_value}")

            # update the critic with last 10 frames
            if len(buffer):
                online_critic_update(model, buffer, num_pos_samples=50, num_neg_samples=100, num_iterations=30, 
                            learning_rate=1e-5, iou_threshold=0.7, batch_size=128, device=device)
        
        
        viz_frame = np.ascontiguousarray(current_frame)
        if viz_frame.shape[2] == 3:
            viz_frame = cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR)
        
        gt_xmin, gt_ymin = int(current_ground_truth[0] - current_ground_truth[2] / 2), int(current_ground_truth[1] - current_ground_truth[3] / 2)
        gt_xmax, gt_ymax = int(current_ground_truth[0] + current_ground_truth[2] / 2), int(current_ground_truth[1] + current_ground_truth[3] / 2)
        cv2.rectangle(viz_frame, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), (0, 255, 0), 2)
        cv2.putText(viz_frame, 'grtr', (gt_xmin, gt_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        
        pred_xmin, pred_ymin = int(current_bbox[0] - current_bbox[2] / 2), int(current_bbox[1] - current_bbox[3] / 2)
        pred_xmax, pred_ymax = int(current_bbox[0] + current_bbox[2] / 2), int(current_bbox[1] + current_bbox[3] / 2)
        cv2.rectangle(viz_frame, (pred_xmin, pred_ymin), (pred_xmax, pred_ymax), (0, 0, 255), 2)
        cv2.putText(viz_frame, 'pred', (pred_xmin, pred_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('actor-critic tracker', viz_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Tracking interrupted by user.")
            break

    cv2.destroyAllWindows()