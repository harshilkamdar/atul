import torch
import torch.nn as nn

from .features import ACTION_DIM, FEATURE_DIM


class AzulNet(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, ACTION_DIM)
        self.win_value_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
        self.margin_head = nn.Linear(hidden_dim, 1)
        self.score_head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        h = self.backbone(x)
        return {
            "policy_logits": self.policy_head(h),
            "win_value": self.win_value_head(h).squeeze(-1),
            "margin": self.margin_head(h).squeeze(-1),
            "scores": self.score_head(h),
        }
