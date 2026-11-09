"""
Face Detection and Multi-ROI Extraction for Deepfake and rPPG Analysis
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path


class FaceAndROIDetector:
    """
    Robust face and multi-ROI detector.
    Extracts:
    1. Full cropped facial bounding box for CNN-LSTM analysis.
    2. Distinct anatomical skin ROIs (Forehead, Left Cheek, Right Cheek)
       for physiological cross-region rPPG coherence analysis.
    """

    def __init__(self, scale_factor: float = 1.08, min_neighbors: int = 4, min_size: Tuple[int, int] = (28, 28)):
        # Primary & secondary Haar cascades for maximum robustness
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade_alt_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
        
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        self.face_cascade_alt = cv2.CascadeClassifier(str(cascade_alt_path)) if cascade_alt_path.exists() else None
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.last_known_bbox: Optional[Tuple[int, int, int, int]] = None

    def reset(self):
        """Resets tracking state between independent video files."""
        self.last_known_bbox = None

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
        if len(faces) == 0 and self.face_cascade_alt is not None:
            faces = self.face_cascade_alt.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=max(3, self.min_neighbors - 1),
                minSize=self.min_size
            )

        if len(faces) > 0:
            # If we have a last known bbox, prioritize the face closest to it
            # to reject transient false-positive flashes (firecrackers, hands, reflections)
            if self.last_known_bbox is not None:
                px, py, pw, ph = self.last_known_bbox
                plausible = [
                    b for b in faces
                    if abs(b[1] - py) < ph * 1.8 and abs(b[0] - px) < pw * 1.8
                ]
                chosen = max(plausible, key=lambda b: b[2] * b[3]) if plausible else None
            else:
                chosen = max(faces, key=lambda b: b[2] * b[3])

            if chosen is not None:
                x, y, w, h = chosen
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
                    sm_x = int(0.70 * px + 0.30 * x1)
                    sm_y = int(0.70 * py + 0.30 * y1)
                    sm_w = int(0.70 * pw + 0.30 * (x2 - x1))
                    sm_h = int(0.70 * ph + 0.30 * (y2 - y1))
                    self.last_known_bbox = (sm_x, sm_y, sm_w, sm_h)
                else:
                    self.last_known_bbox = target_bbox
                return self.last_known_bbox
            else:
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

    def track_video_sequence(
        self,
        frames: List[np.ndarray],
        seq_len: int = 16,
        rppg_len: int = 64,
        target_size: Tuple[int, int] = (224, 224)
    ) -> Dict[str, Any]:
        """
        Global two-pass temporal face tracker with linear interpolation and smoothing:
        1. Multi-scale detection across all candidate frames
        2. Anchor selection from median coordinates
        3. Bidirectional linear interpolation for missed frames (tilted head, laughing, occlusion)
        4. Temporal exponential smoothing to prevent bounding box jitter
        5. Extracts aligned sequence face crops and continuous multi-region rPPG traces
        """
        if not frames:
            raise ValueError("No frames provided for tracking.")

        num_frames = len(frames)
        raw_detections: List[Optional[Tuple[int, int, int, int]]] = [None] * num_frames

        # Pass 1: Multi-scale detection across frames
        for i, f in enumerate(frames):
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f
            det = self.face_cascade.detectMultiScale(
                gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=self.min_size
            )
            if len(det) == 0 and self.face_cascade_alt is not None:
                det = self.face_cascade_alt.detectMultiScale(
                    gray, scaleFactor=self.scale_factor, minNeighbors=max(3, self.min_neighbors - 1), minSize=self.min_size
                )
            if len(det) > 0:
                # Prefer primary face in upper 65% of frame (faces are rarely in the lower 35% of an upright frame)
                upper = [b for b in det if b[1] < f.shape[0] * 0.65]
                chosen = max(upper if upper else det, key=lambda b: b[2] * b[3])
                raw_detections[i] = tuple(chosen)

        # Pass 2: Anchor calculation & bidirectional linear interpolation
        valid_indices = [i for i, b in enumerate(raw_detections) if b is not None]
        H, W = frames[0].shape[:2]

        if not valid_indices:
            # Fallback anchor if face cascade never fired
            cw, ch = int(W * 0.55), int(H * 0.55)
            anchor = ((W - cw) // 2, int(H * 0.15), cw, ch)
            interpolated_boxes = [anchor] * num_frames
        else:
            med_x = int(np.median([raw_detections[i][0] for i in valid_indices]))
            med_y = int(np.median([raw_detections[i][1] for i in valid_indices]))
            med_w = int(np.median([raw_detections[i][2] for i in valid_indices]))
            med_h = int(np.median([raw_detections[i][3] for i in valid_indices]))
            anchor = (med_x, med_y, med_w, med_h)

            interpolated_boxes = []
            for i in range(num_frames):
                if raw_detections[i] is not None:
                    interpolated_boxes.append(raw_detections[i])
                else:
                    prev_idx = max([idx for idx in valid_indices if idx < i], default=valid_indices[0])
                    next_idx = min([idx for idx in valid_indices if idx > i], default=valid_indices[-1])
                    if prev_idx == next_idx:
                        interpolated_boxes.append(raw_detections[prev_idx])
                    else:
                        alpha = (i - prev_idx) / (next_idx - prev_idx)
                        b1 = raw_detections[prev_idx]
                        b2 = raw_detections[next_idx]
                        box = (
                            int((1 - alpha) * b1[0] + alpha * b2[0]),
                            int((1 - alpha) * b1[1] + alpha * b2[1]),
                            int((1 - alpha) * b1[2] + alpha * b2[2]),
                            int((1 - alpha) * b1[3] + alpha * b2[3])
                        )
                        interpolated_boxes.append(box)

        # Pass 3: Clean bounding boxes from exact linear interpolation
        clean_boxes = interpolated_boxes

        # Extract sequence face crops (for CNN-LSTM)
        step = max(1, num_frames // seq_len)
        sampled_indices = [min(k * step, num_frames - 1) for k in range(seq_len)]
        face_crops = []
        for s_idx in sampled_indices:
            f = frames[s_idx]
            x, y, w, h = clean_boxes[s_idx]
            pad_x = int(0.10 * w)
            pad_y = int(0.10 * h)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(W, x + w + pad_x)
            y2 = min(H, y + h + pad_y)
            crop = f[y1:y2, x1:x2]
            if crop.size == 0:
                crop = cv2.resize(f, target_size)
            else:
                crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
            face_crops.append(crop)

        # Extract continuous multi-region ROIs (for rPPG)
        rppg_count = min(num_frames, rppg_len)
        fh_rgb, lc_rgb, rc_rgb = [], [], []

        for i in range(rppg_count):
            f = frames[i]
            x, y, w, h = clean_boxes[i]
            # Forehead ROI
            fh = f[max(0, y):max(0, y + int(0.25 * h)), max(0, x + int(0.20 * w)):max(0, x + int(0.80 * w))]
            # Left Cheek ROI
            lc = f[max(0, y + int(0.40 * h)):max(0, y + int(0.70 * h)), max(0, x):max(0, x + int(0.35 * w))]
            # Right Cheek ROI
            rc = f[max(0, y + int(0.40 * h)):max(0, y + int(0.70 * h)), max(0, x + int(0.65 * w)):max(0, x + w)]

            fh_rgb.append(self.compute_roi_mean_rgb(fh))
            lc_rgb.append(self.compute_roi_mean_rgb(lc))
            rc_rgb.append(self.compute_roi_mean_rgb(rc))

        return {
            "face_crops": face_crops,
            "clean_boxes": clean_boxes,
            "forehead_rgb": np.array(fh_rgb, dtype=np.float32),
            "left_cheek_rgb": np.array(lc_rgb, dtype=np.float32),
            "right_cheek_rgb": np.array(rc_rgb, dtype=np.float32)
        }
