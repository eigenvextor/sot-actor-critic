import os
import csv
import copy
import random
from collections import deque
import torch
import torch.optim as optim
import torch.nn as nn
import utils
from tqdm import tqdm

class ReplayBuffer:
    def __init__(self, capacity=10000, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.initialized = False

    def push(self, state, action, reward, next_state):
        # fill the whole buffer w empty values which will be overwritten
        if not self.initialized:
            self.states = torch.empty((self.capacity, *state.shape[1:]), dtype=state.dtype, device=self.device)
            self.actions = torch.empty((self.capacity, *action.shape[1:]), dtype=action.dtype, device=self.device)
            self.rewards = torch.empty((self.capacity, *reward.shape[1:]), dtype=reward.dtype, device=self.device)
            self.next_states = torch.empty((self.capacity, *next_state.shape[1:]), dtype=next_state.dtype, device=self.device)
            self.initialized = True

        self.states[self.ptr] = state.detach().squeeze(0)
        self.actions[self.ptr] = action.detach().squeeze(0)
        self.rewards[self.ptr] = reward.detach().squeeze(0)
        self.next_states[self.ptr] = next_state.detach().squeeze(0)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs]

    def __len__(self):
        return self.size

def supervised_iteration(model, initial_frame, initial_gt, mse_loss, device):
    # 1e-4 LR specifically for the initialization stage

    actor_params = list(model.actor.parameters())
    init_optimizer = optim.Adam(actor_params, lr=1e-4)
    
    sampled_bboxes = utils.generate_actor_samples(initial_gt, num_samples=32)
    
    state_tensors = []
    target_actions = []
    
    for bbox in sampled_bboxes:
        expert_action = utils.get_expert_action(bbox, initial_gt, device)
        state_tensor = utils.get_state_patch(initial_frame, bbox, device=device)
        
        state_tensors.append(state_tensor)
        target_actions.append(expert_action)
        
    batch_states = torch.cat(state_tensors)
    batch_targets = torch.cat(target_actions)
    
    with torch.no_grad():
        features = model.extract_actor_features(batch_states)
        
    predicted_actions = model.get_action(features)
    loss = mse_loss(predicted_actions, batch_targets)
    
    init_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_params, max_norm=4.0)
    init_optimizer.step()

def train_offline_ddpg(model, train_loader, device, total_iterations=50000, checkpoint_iterations=5000, batch_size=64):

    # create target networks
    target_model = copy.deepcopy(model).to(device)

    # freeze target networks
    for param in target_model.parameters():
        param.requires_grad = False

    target_model.eval()

    # lockin resnet layers to avoid changing batchnorm mean, std
    model.train()
    model.actor_feature_extractor.eval()
    model.critic_feature_extractor.eval()
    
    # Paper LRs: Actor 1e-6, Critic 1e-5
    actor_params = list(model.actor.parameters())
    actor_optimizer = optim.Adam(actor_params, lr=1e-6)

    critic_params = list(model.c_fc1.parameters()) + \
                    list(model.c_fc2.parameters())
    critic_optimizer = optim.Adam(critic_params, lr=1e-5)
    
    mse_loss = nn.MSELoss()

    replay_buffer = ReplayBuffer(capacity=10000, device=device)
    
    epsilon = 0.7
    current_iteration = 0

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    log_file = open('metrics/training_metrics.csv', mode='w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['iteration', 'actor_loss', 'critic_loss', 'epsilon'])

    print("starting offline DDPG training")

    # tqdm progress bar
    pbar = tqdm(total=total_iterations, desc="training ACT tracker", unit="iter")
    
    while current_iteration < total_iterations:
        # go across all videos one by one and select random video sequence
        for frames_list, gt_boxes_list in train_loader:
            if current_iteration >= total_iterations:
                break
            
            frames = [f.squeeze(0).numpy() for f in frames_list]
            # 30 frames chosen always
            # max_T = min(len(frames) - 1, random.randint(20, 40))
            max_T = min(len(frames) - 1, 30)
            
            if max_T < 2:
                continue
                
            initial_frame = frames[0]
            initial_gt = gt_boxes_list[0]
            supervised_iteration(model, initial_frame, initial_gt, mse_loss, device)
            
            current_bbox = initial_gt
            
            # replay buffer data
            for t in range(1, max_T + 1):
                current_frame = frames[t]
                ground_truth = gt_boxes_list[t]
                
                # 1. obtain state s_t
                state_tensor = utils.get_state_patch(current_frame, current_bbox, device=device)
                
                # 2. select action a_t
                with torch.no_grad():
                    state_features = model.extract_actor_features(state_tensor)
                    if random.random() < epsilon:
                        action = utils.get_expert_action(current_bbox, ground_truth, device)
                    else:
                        action = model.get_action(state_features)
                
                # 3. execute action
                next_bbox = utils.apply_action(current_bbox, action)
                next_state_tensor = utils.get_state_patch(current_frame, next_bbox, device=device)
                    
                iou = utils.calculate_iou(next_bbox, ground_truth)

                # # to avoid polluting the buffer
                # if iou < 0.3:
                #     continue

                reward = 1.0 if iou > 0.7 else -1.0
                reward_tensor = torch.tensor([[reward]], dtype=torch.float32, device=device)
                
                # 4. store transition (states are stored not state features)
                replay_buffer.push(state_tensor, action, reward_tensor, next_state_tensor)
                current_bbox = next_bbox

            # sample a random mini-batch
            if len(replay_buffer) >= batch_size:
                for _ in range(25):
                    b_state, b_action, b_reward, b_next_state = replay_buffer.sample(batch_size)

                    # critic loss = 1/n sum(r + gamma*Q'(s', mu'(s'|theta^mu') | theta^Q') - Q(s, a | theta^Q))**2
                    with torch.no_grad():
                        # mu'(s'|theta^mu')
                        b_next_state_features = target_model.extract_actor_features(b_next_state)
                        next_action = target_model.get_action(b_next_state_features)
                        
                        # Q'(s', mu'(s'|theta^mu') | theta^Q')
                        b_next_state_features_c = target_model.extract_critic_features(b_next_state)
                        target_q_val = target_model.get_q_value_offline(b_next_state_features_c, next_action)

                        # r + gamma*Q'(s', mu'(s'|theta^mu') | theta^Q')
                        target_q = b_reward + 0.99 * target_q_val

                    # Q(s, a | theta^Q)
                    b_state_features_c = model.extract_critic_features(b_state)
                    current_q = model.get_q_value_offline(b_state_features_c, b_action)
                    
                    critic_loss = mse_loss(current_q, target_q)
                    
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic_params, max_norm=7.5)
                    critic_optimizer.step()
                    
                    # actor loss = 1/ N sum(Q(s, mu(s | theta^mu)| theta^Q))
                    
                    # mu(s | theta^mu)
                    b_state_features = model.extract_actor_features(b_state)
                    actor_action = model.get_action(b_state_features)

                    # if current_iteration % 1000 == 0: 
                    #     print(f"\nPositive rewards in batch: {(b_reward == 1.0).sum().item()} / {batch_size}")
                    #     print(f"Actor actions max: {actor_action.max().item():.4f}, min: {actor_action.min().item():.4f}")

                    # Q(s, mu(s | theta^mu)| theta^Q)
                    # no backprop for critic

                    for param in critic_params:
                        param.requires_grad = False

                    actor_loss = -model.get_q_value_offline(b_state_features_c.detach(), actor_action).mean()
                    
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_params, max_norm=4.0)
                    actor_optimizer.step()

                    for param in critic_params:
                        param.requires_grad = True

                    current_iteration += 1

                    pbar.update(1)
                
                    if current_iteration % 10 == 0:
                        tau = 0.001
                        with torch.no_grad():
                            for target_param, main_param in zip(target_model.parameters(), model.parameters()):
                                target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

                        # log and update losses every 10 iterations
                        pbar.set_postfix({
                            'A_Loss': f"{actor_loss.item():.4f}", 
                            'C_Loss': f"{critic_loss.item():.4f}",
                            'Eps': f"{epsilon:.2f}"
                        })
                        
                        log_writer.writerow([
                            current_iteration, 
                            actor_loss.item(), 
                            critic_loss.item(), 
                            epsilon
                        ])
                
                    if current_iteration % 100 == 0:
                        log_file.flush()
                
                    if current_iteration % 1000 == 0:
                        epsilon = max(0.05, epsilon * 0.975)

                    if current_iteration % checkpoint_iterations == 0: 
                        checkpoint_path = f"checkpoints/act_tracker_iter_{current_iteration}.pth"
                        torch.save({
                            'iteration': current_iteration,
                            'model': model.state_dict(),
                            'epsilon': epsilon
                        }, checkpoint_path)

    # cleanup after training finishes
    pbar.close()
    log_file.close()
    print("offline training complete")