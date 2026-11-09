"""
Bidirectional LSTM with Temporal Self-Attention for Video Sequence Forensics
Analyzes inter-frame consistency and temporal anomalies (flickering, unnatural blinks).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from deepfake_system.config import (
    CNN_FEATURE_DIM, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT, NUM_CLASSES
)


class TemporalSelfAttention(nn.Module):
    """
    Computes attention weights over frame sequence timesteps.
    Highlights specific anomaly frames (e.g., eye blink glitch or mouth warping).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: tensor of shape (B, T, hidden_dim)
        Returns:
            context: (B, hidden_dim) attention-weighted summary vector
            attn_weights: (B, T) normalized frame importance scores
        """
        # energy shape: (B, T, 1)
        energy = self.projection(h)
        attn_weights = F.softmax(energy, dim=1).squeeze(-1)  # (B, T)
        # Context vector: sum(attn_weights * h)
        context = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # (B, hidden_dim)
        return context, attn_weights


class VideoTemporalLSTM(nn.Module):
    """
    Sequence-level temporal forensic network:
    Input: (B, T, CNN_FEATURE_DIM)
    Processing: 2-Layer Bidirectional LSTM + Temporal Self-Attention
    Output: (B, LSTM_HIDDEN_DIM * 2) dense temporal representation
    """

    def __init__(
        self,
        input_dim: int = CNN_FEATURE_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        num_classes: int = NUM_CLASSES
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.bidir_dim = hidden_dim * 2
        self.attention = TemporalSelfAttention(self.bidir_dim)

        # Standalone classification head (for CNN+LSTM evaluation mode)
        self.classifier = nn.Sequential(
            nn.Linear(self.bidir_dim, self.bidir_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.bidir_dim // 2, num_classes)
        )

    def forward_features(self, sequence_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sequence_features: (B, T, input_dim)
        Returns:
            temporal_vector: (B, bidir_dim)
            frame_attention_weights: (B, T)
        """
        x = self.input_norm(sequence_features)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)  # (B, T, bidir_dim)
        context, attn_weights = self.attention(lstm_out)
        return context, attn_weights

    def forward(self, sequence_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, num_classes)
            frame_attention_weights: (B, T)
        """
        context, attn_weights = self.forward_features(sequence_features)
        logits = self.classifier(context)
        return logits, attn_weights
