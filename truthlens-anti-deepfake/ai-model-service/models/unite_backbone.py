# models/unite_backbone.py
import torch.nn as nn

class UniteBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Placeholder for transformer-based backbone
        self.layer = nn.Linear(128, 64)
    
    def forward(self, x):
        return self.layer(x)
