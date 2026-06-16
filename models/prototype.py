import torch
import torch.nn as nn

class Prototype(nn.Module):
    def __init__(self, num_actions: int = 8, embed_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(16, embed_dim)  # 16 ARC colors
        self.cnn = nn.Sequential(
            nn.Conv2d(embed_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, num_actions)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # grid: (B, 64, 64) integer color ids
        x = self.embed(grid)              # (B, 64, 64, embed_dim)
        x = x.permute(0, 3, 1, 2)        # (B, embed_dim, 64, 64)
        x = self.cnn(x).flatten(1)       # (B, 64)
        return self.head(x)              # (B, 8) logits