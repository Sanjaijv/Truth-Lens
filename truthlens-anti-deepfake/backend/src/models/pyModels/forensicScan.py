import torch
import torch.nn as nn
import timm
import cv2
import numpy as np

class ForensicScanModel(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0
        )
        self.feat_dim = self.backbone.num_features

        self.physics_encoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 3)
        )
    def forward(self, frames):
        B, T, C, H, W = frames.shape

        x = frames.reshape(B * T, C, H, W)
        vit_feats = self.backbone(x)
        vit_feats = vit_feats.view(B, T, -1).mean(dim=1)

        physics_feats = torch.rand(B, 6).to(frames.device)  # placeholder for demo
        physics_enc = self.physics_encoder(physics_feats)

        fused = torch.cat([vit_feats, physics_enc], dim=1)
        logits = self.classifier(fused)

        return logits

