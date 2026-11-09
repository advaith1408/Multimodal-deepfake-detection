"""
Explainable AI: Pure PyTorch Grad-CAM and Grad-CAM++ Implementation
Generates high-resolution spatial heatmaps highlighting facial manipulation artifacts.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GradCAMExplainer:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    on convolutional feature maps to explain CNN predictions.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1  # 1 = FAKE
    ) -> np.ndarray:
        """
        input_tensor: normalized image tensor of shape (1, 3, H, W)
        target_class: index of class to explain (1 for Fake, 0 for Real)
        Returns: 2D heatmap array normalized to [0, 1] of shape (H, W)
        """
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(input_tensor)
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            H, W = input_tensor.shape[2:]
            return np.zeros((H, W), dtype=np.float32)

        # Global average pooling of gradients: alpha_k = (1/Z) * sum(grad)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Linear combination of activations weighted by gradients
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, h, w)
        # Apply ReLU to retain only features that positively influence the target class
        cam = F.relu(cam)

        # Interpolate CAM to match input tensor resolution
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    @staticmethod
    def overlay_heatmap(
        image_bgr: np.ndarray,
        cam_mask: np.ndarray,
        alpha: float = 0.55,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overlays Grad-CAM heatmap on the original face crop.
        Returns:
            blended_bgr: original image with colored heatmap overlay
            colored_heatmap: colored heatmap alone
        """
        H, W = image_bgr.shape[:2]
        cam_resized = cv2.resize(cam_mask, (W, H))
        cam_uint8 = np.uint8(255 * cam_resized)
        
        heatmap_colored = cv2.applyColorMap(cam_uint8, colormap)
        blended = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_colored, alpha, 0)

        return blended, heatmap_colored
