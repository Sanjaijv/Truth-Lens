import torch
import torch.nn as nn
import timm
import cv2
import numpy as np

class DeepScanModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0
        )
        self.feat_dim = self.backbone.num_features

        self.physics_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
    def _physics_features(self,f1,f2):
        gray1 = cv2.cvtColor((f1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor((f2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        noise = cv2.Laplacian(gray1, cv2.CV_64F).var()
        brightness = gray1.mean() / 255.0
        stability = np.exp(-np.abs(gray1 - gray2).mean() / 50)
        sharpness = gray1.var() / 10000

        return torch.tensor([
            noise / 1000,
            brightness,
            stability,
            sharpness
        ])
    def forward(self, frames):
        B, T, C, H, W = frames.shape

        x = frames.reshape(B * T, C, H, W)
        vit_feats = self.backbone(x)
        vit_feats = vit_feats.view(B, T, -1).mean(dim=1)

        physics = []
        for b in range(B):
            feats = []
            for t in range(min(T-1,5)):
                f1=frames[b, t].cpu().numpy().transpose(1, 2, 0)
                f2=frames[b, t + 1].cpu().numpy().transpose(1, 2, 0)
                feats.append(self.__physics_features(f1,f2))
            physics.append(torch.stack(feats).mean(dim=0))
        physics = torch.stack(physics).to(frames.device)
        physics_enc=self.physics_encoder(physics)

        fused=torch.cat([vit_feats, physics_enc], dim=1)
        logits=self.classifier(fused)

        return logits.squeeze(1)