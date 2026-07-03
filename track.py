# have to load model checkpoints
# need to pass the data thr model
# initialization strategy
# actor: m candidate boxes are generated around the ground truth values and calculate the accurate actions 
# that needs to be taken. make it as a regression problem and minimize l2 loss 
# critic: m candidate boxes are generated around the ground truth values and assign labels depending on overlap
# ratio. make it as a binary classifier problem and minimize binary cross entropy loss 
# redetection strategy: if critic value < 0
# samples are drawn around previous bounding box predictions. the one with highest critic score is chosen
# critic update: same as critic initialization but using last 10 frames 

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
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

def initialize_actor(model, initial_frame, initial_gt, num_samples=32, num_iterations=5, learning_rate=1e-4, iou_threshold=0.7, device="cpu"):
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
        
    batch_states = torch.cat(state_tensors)
    batch_targets = torch.cat(target_actions)
    
    with torch.no_grad():
        features = model.extract_actor_features(batch_states)
    
    for i in range(num_iterations):
        predicted_actions = model.get_action(features)
        loss = mse_loss(predicted_actions, batch_targets)
        
        init_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_params, max_norm=4.0)
        init_optimizer.step()

        print(f"iteration {i}: {loss.item():.4f}")

def initialize_critic(model, initial_frame, initial_gt, num_pos_samples=50, num_neg_samples=100, num_iterations=5, learning_rate=1e-4, iou_threshold=0.7, device="cpu"):
    # 1e-4 LR specifically for the initialization stage
    critic_params = list(model.critic.parameters())
    init_optimizer = optim.Adam(critic_params, lr=learning_rate)
    
    bce_loss = nn.BCEWithLogitsLoss()

    pos_samples, neg_samples = utils.generate_critic_samples(initial_gt, num_pos_samples=num_pos_samples, num_neg_samples=num_neg_samples, iou_threshold=iou_threshold)

    

def online_tracking(model, frames, initial_gt, device):
    
    model.train()
    model.actor_feature_extractor.eval()
    model.critic_feature_extractor.eval()
    
    # actor initialization
    frame_0 = frames[0]
    initialize_actor(model, frame_0, initial_gt, num_samples=32, num_iterations=5, 
                     learning_rate=1e-4, iou_threshold=0.7, device=device)
    
    # critic initialization
    initialize_critic(model, frame_0)

    # subsequent frames are processed
    model.eval()

    current_bbox = initial_gt
    for current_frame in frames[1:]:
        
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
        else:
            # apply redetection strategy
            pass
        
