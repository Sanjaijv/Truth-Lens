import torch
import torch.nn as nn
import timm

class QuickScanModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0
        )
        self.feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    def forward(self, frames):
        B, T, C, H, W = frames.shape
        sampled = frames[:, ::2]
        B, Ts, C, H, W = sampled.shaped
        x = sampled.reshape(B * Ts, C, H, W)
        feats = self.backbone(x)
        feats = feats.view(B, Ts, -1).mean(dim=1)

        logits = self.classifier(feats)
        return logits.squeeze(1)
