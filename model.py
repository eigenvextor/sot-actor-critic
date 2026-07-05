import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class ActorCriticTracker(nn.Module):
    def __init__(self, freeze_backbone=True, device="mps"):
        super(ActorCriticTracker, self).__init__()
        self.device = device

        resnet_actor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.actor_feature_extractor = nn.Sequential(*list(resnet_actor.children())[:-3])

        resnet_critic = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.critic_feature_extractor = nn.Sequential(*list(resnet_critic.children())[:-3])
        
        if freeze_backbone:
            # freeze all resnet18 layers
            for param in self.actor_feature_extractor.parameters():
                param.requires_grad = False
            for param in self.critic_feature_extractor.parameters():
                param.requires_grad = False
                
            # unfreeze only the last layer (layer3) for both actor critic
            for child in list(self.actor_feature_extractor.children())[-1:]:
                for param in child.parameters():
                    param.requires_grad = True
            for child in list(self.critic_feature_extractor.children())[-1:]:
                for param in child.parameters():
                    param.requires_grad = True
        
        # FIXED while loading data: input patch of 107x107
        self.feature_dim = 256 * 7 * 7
        
        self.actor = nn.Sequential(
            nn.Flatten(), # 12,544
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Tanh()
        )

        self.c_fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )

        self.c_fc2 = nn.Sequential(
            nn.Linear(512+4, 1)
        )


    def extract_actor_features(self, state_patch):
        return self.actor_feature_extractor(state_patch)
    
    def extract_critic_features(self, state_patch):
        return self.critic_feature_extractor(state_patch)

    def get_action(self, state_features):
        action = self.actor(state_features)
        # have to multiply with another tensor so that computational graph doesnt break
        scale_factors = torch.tensor([1.0, 1.0, 0.05, 0.05], device=self.device)
        return action * scale_factors
    
    def get_q_value_offline(self, state_features, action):
        f_out = self.c_fc1(state_features)
        combined = torch.cat([f_out, action], dim=1)
        return self.c_fc2(combined)

    def get_confidence_value_online(self, state_features):
        f_out = self.c_fc1(state_features)
        weights = self.c_fc2[0].weight[:, :512]
        bias = self.c_fc2[0].bias
        return F.linear(f_out, weights, bias)