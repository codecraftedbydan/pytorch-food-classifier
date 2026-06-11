"""
Reusable FoodClassifier model module.
Imported by the training script, evaluation, and Grad-CAM.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B2_Weights


class FoodClassifier(nn.Module):
    """
    EfficientNet-B2 backbone + custom classification head for Food-101.

    Args:
        num_classes      : number of output classes (101 for Food-101)
        freeze_backbone  : if True, backbone weights are frozen on init
    """

    def __init__(self, num_classes: int = 101, freeze_backbone: bool = True):
        super().__init__()

        backbone = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

        self.features  = backbone.features
        self.avgpool   = backbone.avgpool
        in_features    = backbone.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_last_n_blocks: int = 3):
        for param in self.features.parameters():
            param.requires_grad = False
        for block in list(self.features.children())[-unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfrozen last {unfreeze_last_n_blocks} blocks. "
              f"Trainable params: {trainable:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
