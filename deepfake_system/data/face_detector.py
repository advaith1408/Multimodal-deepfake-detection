"""
Face Detection and Multi-ROI Extraction for Deepfake and rPPG Analysis
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path


class FaceAndROIDetector:
    """
    Robust face and multi-ROI detector.
    Extracts:
    1. Full cropped facial bounding box for CNN-LSTM analysis.
    2. Distinct anatomical skin ROIs (Forehead, Left Cheek, Right Cheek)
       for physiological cross-region rPPG coherence analysis.
    """

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: Tuple[int, int] = (60, 60)):
        # Initialize OpenCV Haar Cascade
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
        
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.last_known_bbox: Optional[Tuple[int, int, int, int]] = None

    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects primary face in frame. Returns (x, y, w, h) bounding box.
        Employs temporal interpolation fallback if face is momentarily missed.
        """
        if frame is None or frame.size == 0:
            return self.last_known_bbox

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        if len(faces) > 0:
            # Pick the largest face detected by area
            largest_face = max(faces, key=lambda b: b[2] * b[3])
            x, y, w, h = largest_face
            # Expand bounding box slightly (10%) to capture facial boundary blending artifacts
            pad_x = int(0.08 * w)
            pad_y = int(0.08 * h)
            H, W = frame.shape[:2]
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(W, x + w + pad_x)
            y2 = min(H, y + h + pad_y)
            target_bbox = (x1, y1, x2 - x1, y2 - y1)
            if self.last_known_bbox is not None:
                px, py, pw, ph = self.last_known_bbox
                # Apply exponential smoothing if face is the same subject to eliminate crop jitter
                if abs(px - x1) < w * 0.4 and abs(py - y1) < h * 0.4:
                    sm_x = int(0.70 * px + 0.30 * x1)
                    sm_y = int(0.70 * py + 0.30 * y1)
                    sm_w = int(0.70 * pw + 0.30 * (x2 - x1))
                    sm_h = int(0.70 * ph + 0.30 * (y2 - y1))
                    self.last_known_bbox = (sm_x, sm_y, sm_w, sm_h)
                else:
                    self.last_known_bbox = target_bbox
            else:
                self.last_known_bbox = target_bbox
            return self.last_known_bbox

        # Temporal fallback to maintain sequence continuity
        if self.last_known_bbox is not None:
            return self.last_known_bbox

        # Extreme fallback: Center 60% crop of frame
        H, W = frame.shape[:2]
        cw, ch = int(W * 0.6), int(H * 0.6)
        cx, cy = (W - cw) // 2, (H - ch) // 2
        return (cx, cy, cw, ch)

    def crop_face(self, frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Crops primary face and resizes to target_size (width, height).
        """
        bbox = self.detect_face(frame)
        if bbox is None:
            return cv2.resize(frame, target_size)

        x, y, w, h = bbox
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            return cv2.resize(frame, target_size)

        return cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)

    def extract_rois(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extracts anatomical sub-regions from the face:
        - Forehead ROI: Glabella and upper frontalis (vascularized by supraorbital artery)
        - Left Cheek ROI: Zygomatic / malar region (facial artery)
        - Right Cheek ROI: Zygomatic / malar region
        Returns dictionary of ROI image patches.
        """
        bbox = self.detect_face(frame)
        if bbox is None:
            H, W = frame.shape[:2]
            bbox = (0, 0, W, H)

        x, y, w, h = bbox
        face_patch = frame[y:y+h, x:x+w]
        fh, fw = face_patch.shape[:2]

        if fh < 10 or fw < 10:
            face_patch = cv2.resize(frame, (100, 100))
            fh, fw = 100, 100

        # Anatomical coordinates relative to face bounding box
        # Forehead: Y: 15% to 32%, X: 30% to 70%
        forehead = face_patch[int(fh * 0.15):int(fh * 0.32), int(fw * 0.30):int(fw * 0.70)]
        # Left Cheek: Y: 50% to 72%, X: 22% to 42%
        left_cheek = face_patch[int(fh * 0.50):int(fh * 0.72), int(fw * 0.22):int(fw * 0.42)]
        # Right Cheek: Y: 50% to 72%, X: 58% to 78%
        right_cheek = face_patch[int(fh * 0.50):int(fh * 0.72), int(fw * 0.58):int(fw * 0.78)]

        return {
            "forehead": forehead if forehead.size > 0 else face_patch,
            "left_cheek": left_cheek if left_cheek.size > 0 else face_patch,
            "right_cheek": right_cheek if right_cheek.size > 0 else face_patch,
            "face_crop": face_patch,
            "bbox": np.array([x, y, w, h], dtype=np.int32)
        }

    def compute_roi_mean_rgb(self, roi: np.ndarray) -> np.ndarray:
        """
        Calculates spatial mean RGB values across skin pixels within an ROI.
        Ignores extreme dark/specular saturation pixels.
        Returns shape (3,) array: [R_mean, G_mean, B_mean].
        """
        if roi is None or roi.size == 0:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # Intensity filter to reject non-skin specular reflection or deep shadow
        intensity = np.mean(rgb, axis=2)
        valid_mask = (intensity > 25) & (intensity < 240)
        
        if np.sum(valid_mask) > 10:
            r_mean = np.mean(rgb[:, :, 0][valid_mask])
            g_mean = np.mean(rgb[:, :, 1][valid_mask])
            b_mean = np.mean(rgb[:, :, 2][valid_mask])
        else:
            r_mean = np.mean(rgb[:, :, 0])
            g_mean = np.mean(rgb[:, :, 1])
            b_mean = np.mean(rgb[:, :, 2])

        return np.array([r_mean, g_mean, b_mean], dtype=np.float32)
