"""
Gated Multimodal Fusion (GMU) and Human-in-the-Loop Switching
Combines Spatial-Temporal (CNN-LSTM) and Physiological (rPPG) Representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from deepfake_system.config import (
    LSTM_HIDDEN_DIM, RPPG_FEATURE_DIM, FUSION_HIDDEN_DIM, NUM_CLASSES,
    MODE_STANDARD, MODE_ADVANCED
)


class GatedMultimodalFusion(nn.Module):
    """
    Canonical Gated Multimodal Fusion Unit (GMU, Arevalo et al.).
    Computes dynamic modality gating between Spatial-Temporal (CNN-LSTM) and Physiological (rPPG).
    Supports Human-in-the-Loop switching between Standard (CNN+LSTM) and Advanced (CNN+LSTM+rPPG).
    """

    def __init__(
        self,
        st_dim: int = LSTM_HIDDEN_DIM * 2,  # 512
        rppg_dim: int = RPPG_FEATURE_DIM,   # 32
        fusion_dim: int = FUSION_HIDDEN_DIM,# 256
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.2
    ):
        super().__init__()
        self.st_dim = st_dim
        self.rppg_dim = rppg_dim
        self.fusion_dim = fusion_dim

        # Canonical Modality Projections: h = tanh(W x + b)
        self.st_proj = nn.Sequential(
            nn.Linear(st_dim, fusion_dim),
            nn.Tanh()
        )

        self.rppg_proj = nn.Sequential(
            nn.Linear(rppg_dim, fusion_dim),
            nn.Tanh()
        )

        # Dynamic Gating Network: z = sigmoid(W_z [h_st, h_rppg])
        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )

        # Final Linear Classification Head
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(
        self,
        h_st: torch.Tensor,
        h_rppg: Optional[torch.Tensor] = None,
        mode: str = MODE_ADVANCED
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with modal switching:
        h_st: (B, st_dim)
        h_rppg: (B, rppg_dim) - optional in standard mode
        mode: 'standard' or 'advanced'

        Returns:
            dict containing:
            - logits: (B, num_classes)
            - probabilities: (B, num_classes)
            - gate_weight: mean gate score (modality importance)
            - fused_representation: (B, fusion_dim)
        """
        e_st = self.st_proj(h_st)  # (B, fusion_dim)

        if mode == MODE_STANDARD or h_rppg is None:
            # Standard Mode: pure Spatial-Temporal (CNN-LSTM)
            fused = e_st
            gate_val = torch.ones((h_st.size(0), 1), device=h_st.device)
        else:
            # Advanced Mode: Gated Multimodal Fusion with Physiological rPPG
            e_rppg = self.rppg_proj(h_rppg)  # (B, fusion_dim)
            concat_modalities = torch.cat([e_st, e_rppg], dim=-1)
            gate = self.gate_net(concat_modalities)  # (B, fusion_dim) in (0, 1)
            # Fused representation: z * e_st + (1 - z) * e_rppg
            fused = gate * e_st + (1.0 - gate) * e_rppg
            gate_val = torch.mean(gate, dim=-1, keepdim=True)

        logits = self.classifier(self.dropout(fused))
        probabilities = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "gate_weight": gate_val,
            "fused_representation": fused
        }
