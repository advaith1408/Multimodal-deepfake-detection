"""
Global Configuration and Hyperparameters
Explainable Multimodal Deepfake Face Detection (CNN + LSTM + rPPG)
"""

import os
import tempfile
from pathlib import Path
import torch

# Configure Matplotlib writable cache directory
mpl_dir = Path(__file__).resolve().parent / ".mplconfig"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

# Base directories
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True, parents=True)

# Hardware execution
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Safe default on Windows

# Video Sequence Preprocessing
DEFAULT_SEQUENCE_LENGTH = 16       # Number of frames fed into LSTM
RPPG_WINDOW_LENGTH = 64           # Number of frames required for stable rPPG FFT (approx 2 sec at 30 fps)
FRAME_SIZE = (224, 224)            # Width, Height for CNN input
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# rPPG Biological Parameters
FPS_DEFAULT = 30.0                 # Assumed video frame rate if metadata is absent
RPPG_LOW_FREQ = 0.7                # 42 BPM (low human heart rate threshold)
RPPG_HIGH_FREQ = 3.5               # 210 BPM (high human heart rate threshold)
FILTER_ORDER = 3                   # Butterworth bandpass filter order

# Model Dimensionality
CNN_BACKBONE_NAME = "efficientnet_b0"
CNN_FEATURE_DIM = 1280             # EfficientNet-B0 penultimate feature dim (1792 for B4)
LSTM_HIDDEN_DIM = 256              # Bi-LSTM hidden state dimension (Bidirectional -> 512)
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
RPPG_FEATURE_DIM = 32              # Statistical + Spectral descriptors from multi-region rPPG
FUSION_HIDDEN_DIM = 256
NUM_CLASSES = 2                    # 0: Real, 1: Fake

# Human-in-the-Loop Modes
MODE_STANDARD = "standard"         # CNN + LSTM (Fast, spatial-temporal)
MODE_ADVANCED = "advanced"         # CNN + LSTM + rPPG (Forensic physiological fusion)

# Explainability Parameters
GRADCAM_TARGET_LAYER = "features.8" # Final convolutional block for EfficientNet-B0
HEATMAP_ALPHA = 0.55
