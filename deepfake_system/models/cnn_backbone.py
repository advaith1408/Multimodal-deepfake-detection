"""
Pretrained CNN Spatial Feature Extractor for Face Manipulation Detection
Utilizes EfficientNet-B0 (or ResNet-50) with Grad-CAM hook capabilities.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
from pathlib import Path

from deepfake_system.config import CNN_FEATURE_DIM, NUM_CLASSES, DEVICE


class CNNSpatialExtractor(nn.Module):
    """
    Spatial feature extractor based on EfficientNet-B0.
    Identifies high-frequency spatial artifacts (blending boundaries, GAN textures).
    """

    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = False, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "efficientnet_b0":
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            in_features = base_model.classifier[1].in_features  # 1280
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, num_classes)
            )
            self.target_conv_layer = self.features[-1]  # features[8]
            self.feature_dim = in_features
        elif backbone_name == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.features = nn.Sequential(
                base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool,
                base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4
            )
            self.avgpool = base_model.avgpool
            in_features = base_model.fc.in_features  # 2048
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes)
            )
            self.target_conv_layer = base_model.layer4[-1]
            self.feature_dim = in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts spatial feature representation.
        x: tensor of shape (B, 3, H, W)
        Returns: tensor of shape (B, feature_dim)
        """
        feat = self.features(x)
        pooled = self.avgpool(feat)
        flat = torch.flatten(pooled, 1)
        return flat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard classification forward pass.
        Returns: logits of shape (B, NUM_CLASSES)
        """
        emb = self.forward_features(x)
        logits = self.classifier(emb)
        return logits

    def freeze_backbone(self):
        """Freezes feature extractor layers for staged transfer learning."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes feature extractor layers for fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

    def save_weights(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: Path, map_location: Optional[torch.device] = None):
        if map_location is None:
            map_location = DEVICE
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state_dict)
