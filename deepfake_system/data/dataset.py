"""
Dataset Management and Temporal Sequence Loading
Handles FaceForensics++ directory structures and temporal sequence sampling.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset

from deepfake_system.config import (
    DEFAULT_SEQUENCE_LENGTH, RPPG_WINDOW_LENGTH, FRAME_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD
)
from deepfake_system.data.face_detector import FaceAndROIDetector
from deepfake_system.models.rppg_extractor import RPPGExtractor


class DeepfakeVideoDataset(Dataset):
    """
    Video sequence dataset for multimodal deepfake detection.
    Reads video files, extracts cropped face sequences, and computes rPPG features.
    """

    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        seq_len: int = DEFAULT_SEQUENCE_LENGTH,
        extract_rppg: bool = True
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.seq_len = seq_len
        self.extract_rppg = extract_rppg
        
        self.face_detector = FaceAndROIDetector()
        self.rppg_extractor = RPPGExtractor()
        
        self.mean = np.array(NORMALIZE_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(NORMALIZE_STD, dtype=np.float32).reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_video_frames(self, path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if len(frames) >= 120:  # Cap maximum read frames
                break
        cap.release()
        return frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._load_video_frames(video_path)
        if len(frames) == 0:
            # Fallback black frames
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.seq_len

        # Sample T frames uniformly
        step = max(1, len(frames) // self.seq_len)
        sampled_indices = [min(i * step, len(frames) - 1) for i in range(self.seq_len)]

        seq_tensors = []
        for s_idx in sampled_indices:
            frame = frames[s_idx]
            crop = self.face_detector.crop_face(frame, target_size=FRAME_SIZE)
            # Normalize to PyTorch (3, H, W)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            norm_crop = (crop_rgb - self.mean) / self.std
            tensor_frame = torch.from_numpy(norm_crop).permute(2, 0, 1).float()
            seq_tensors.append(tensor_frame)

        sequence_tensor = torch.stack(seq_tensors, dim=0)  # (T, 3, 224, 224)

        # rPPG feature vector
        if self.extract_rppg:
            fh_rgb, lc_rgb, rc_rgb = [], [], []
            rppg_frames = frames[:min(len(frames), RPPG_WINDOW_LENGTH)]
            for rf in rppg_frames:
                rois = self.face_detector.extract_rois(rf)
                fh_rgb.append(self.face_detector.compute_roi_mean_rgb(rois["forehead"]))
                lc_rgb.append(self.face_detector.compute_roi_mean_rgb(rois["left_cheek"]))
                rc_rgb.append(self.face_detector.compute_roi_mean_rgb(rois["right_cheek"]))
            
            rppg_vec = self.rppg_extractor.extract_feature_vector(
                np.array(fh_rgb, dtype=np.float32),
                np.array(lc_rgb, dtype=np.float32),
                np.array(rc_rgb, dtype=np.float32)
            )
            rppg_tensor = torch.from_numpy(rppg_vec).float()
        else:
            rppg_tensor = torch.zeros(32, dtype=torch.float32)

        return sequence_tensor, rppg_tensor, label


def create_splits_from_directory(
    root_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict[str, Dict[str, List]]:
    """
    Partitions FaceForensics++ videos into Train, Val, Test without ID leakage.
    Assumes standard FF++ layout:
      root_dir/original_sequences/youtube/c23/videos/*.mp4  (Real, label 0)
      root_dir/manipulated_sequences/Deepfakes/c23/videos/*.mp4 (Fake, label 1)
      root_dir/manipulated_sequences/Face2Face/c23/videos/*.mp4
      root_dir/manipulated_sequences/FaceSwap/c23/videos/*.mp4
      root_dir/manipulated_sequences/NeuralTextures/c23/videos/*.mp4
    """
    root = Path(root_dir)
    real_paths = list(root.glob("original_sequences/**/videos/*.mp4")) + list(root.glob("real/*.mp4"))
    fake_paths = list(root.glob("manipulated_sequences/**/videos/*.mp4")) + list(root.glob("fake/*.mp4"))

    # Extract distinct video IDs to prevent data leakage
    def get_id(p: Path):
        # e.g., 042_089.mp4 -> ID is '042'
        name = p.stem
        return name.split("_")[0] if "_" in name else name

    all_ids = sorted(list(set([get_id(p) for p in real_paths + fake_paths])))
    np.random.seed(42)
    np.random.shuffle(all_ids)

    n_total = len(all_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = set(all_ids[:n_train])
    val_ids = set(all_ids[n_train:n_train + n_val])
    test_ids = set(all_ids[n_train + n_val:])

    splits = {
        "train": {"videos": [], "labels": []},
        "val": {"videos": [], "labels": []},
        "test": {"videos": [], "labels": []}
    }

    for p in real_paths:
        vid_id = get_id(p)
        if vid_id in train_ids:
            splits["train"]["videos"].append(str(p)); splits["train"]["labels"].append(0)
        elif vid_id in val_ids:
            splits["val"]["videos"].append(str(p)); splits["val"]["labels"].append(0)
        else:
            splits["test"]["videos"].append(str(p)); splits["test"]["labels"].append(0)

    for p in fake_paths:
        vid_id = get_id(p)
        if vid_id in train_ids:
            splits["train"]["videos"].append(str(p)); splits["train"]["labels"].append(1)
        elif vid_id in val_ids:
            splits["val"]["videos"].append(str(p)); splits["val"]["labels"].append(1)
        else:
            splits["test"]["videos"].append(str(p)); splits["test"]["labels"].append(1)

    return splits
