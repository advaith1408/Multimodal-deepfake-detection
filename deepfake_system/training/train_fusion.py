"""
Multimodal Model Training and Fine-Tuning Pipeline
Implements Focal Loss, Staged Optimization, and Model Checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from pathlib import Path

from deepfake_system.config import (
    CHECKPOINTS_DIR, DEVICE, NUM_CLASSES, MODE_ADVANCED
)
from deepfake_system.models.deepfake_detector import MultimodalDeepfakeDetector


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for hard example mining in deepfake detection:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples and forces network to focus on ambiguous boundary artifacts.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()


def train_epoch(
    model: MultimodalDeepfakeDetector,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mode: str = MODE_ADVANCED
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for seq_tensors, rppg_tensors, labels in dataloader:
        seq_tensors = seq_tensors.to(DEVICE)    # (B, T, 3, 224, 224)
        rppg_tensors = rppg_tensors.to(DEVICE)  # (B, 32)
        labels = labels.to(DEVICE)              # (B,)

        optimizer.zero_grad()
        B, T, C, H, W = seq_tensors.shape

        # 1. CNN Feature extraction
        frames_flat = seq_tensors.view(B * T, C, H, W)
        spatial_feats = model.cnn.forward_features(frames_flat)
        spatial_seq = spatial_feats.view(B, T, -1)

        # 2. LSTM Temporal analysis
        h_st, _ = model.lstm.forward_features(spatial_seq)

        # 3. Gated Multimodal Fusion
        out = model.fusion(h_st, rppg_tensors, mode=mode)
        logits = out["logits"]

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += B

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1)
    }


def validate_epoch(
    model: MultimodalDeepfakeDetector,
    dataloader: DataLoader,
    criterion: nn.Module,
    mode: str = MODE_ADVANCED
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for seq_tensors, rppg_tensors, labels in dataloader:
            seq_tensors = seq_tensors.to(DEVICE)
            rppg_tensors = rppg_tensors.to(DEVICE)
            labels = labels.to(DEVICE)

            B, T, C, H, W = seq_tensors.shape
            frames_flat = seq_tensors.view(B * T, C, H, W)
            spatial_feats = model.cnn.forward_features(frames_flat)
            spatial_seq = spatial_feats.view(B, T, -1)

            h_st, _ = model.lstm.forward_features(spatial_seq)
            out = model.fusion(h_st, rppg_tensors, mode=mode)
            logits = out["logits"]

            loss = criterion(logits, labels)
            total_loss += loss.item() * B
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += B

    return {
        "val_loss": total_loss / max(total, 1),
        "val_accuracy": correct / max(total, 1)
    }
