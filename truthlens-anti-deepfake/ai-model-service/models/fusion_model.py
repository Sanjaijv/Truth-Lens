# models/fusion_model.py
import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.classifier(x)
