"""
Synthetic Benchmark Video Generator for Offline Validation and Multi-Artifact Simulation
Generates:
1. Authentic human videos: 3D illuminated face, realistic skin micro-texture (survives MP4 compression),
   and coherent hemodynamic hemoglobin pulse (1.15 Hz).
2. Modern Generative AI videos (Diffusion/Sora/Runway style): Over-smoothed skin (low Laplacian variance),
   zero biological cardiac pulse.
3. AI Talking-Head videos (SadTalker/Wav2Lip style): Mouth patch warping, regional chrominance shift.
4. Deepfake FaceSwaps: Outer blending boundary seam, phase desynchronization.
"""

import cv2
import numpy as np
from pathlib import Path


def generate_benchmark_video(
    output_path: str,
    is_fake: bool = False,
    fake_type: str = "diffusion",  # "diffusion", "talking_head", "faceswap"
    duration_sec: float = 2.5,
    fps: int = 30,
    width: int = 320,
    height: int = 320
) -> str:
    """
    Synthesizes a realistic test video.
    """
    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    total_frames = int(duration_sec * fps)

    # Base skin tone in RGB (warm natural human skin)
    base_r, base_g, base_b = 215, 168, 140
    pulse_freq_hz = 1.15  # ~69 BPM (normal resting human heart rate)

    center_x, center_y = width // 2, height // 2
    rx = int(width * 0.35)
    ry = int(rx * 1.25)

    y, x = np.ogrid[:height, :width]

    # Persistent skin texture map (natural micro-texture that survives video compression)
    np.random.seed(int(sum(ord(c) for c in output_path) % 10000))
    raw_tex = np.random.normal(0, 11.5, (height, width, 3)).astype(np.float32)

    for i in range(total_frames):
        t = i / fps
        frame = np.full((height, width, 3), 42, dtype=np.uint8)  # Studio dark backdrop

        # Natural organic head micro-motion
        dx = int(2.5 * np.sin(2 * np.pi * 0.25 * t))
        dy = int(1.8 * np.cos(2 * np.pi * 0.20 * t))
        cx = center_x + dx
        cy = center_y + dy

        # Elliptical 3D coordinate mask
        norm_dist = ((x - cx) / rx)**2 + ((y - cy) / ry)**2
        face_mask = norm_dist <= 1.0
        z_depth = np.sqrt(np.maximum(0, 1.0 - norm_dist))

        # 3D spherical diffuse illumination
        shade = 0.58 + 0.42 * z_depth

        if not is_fake:
            # REAL HUMAN:
            # Subtle physiological blood volume pulse (green channel absorbs hemoglobin)
            pulse = 4.2 * np.sin(2 * np.pi * pulse_freq_hz * t)
            cur_r = base_r + pulse * 0.25
            cur_g = base_g + pulse * 1.10
            cur_b = base_b + pulse * 0.20

            # Real skin: 3D lighting + optical skin micro-texture + sensor grain
            skin = np.zeros((height, width, 3), dtype=np.float32)
            skin[:, :, 0] = cur_b * shade + raw_tex[:, :, 0]
            skin[:, :, 1] = cur_g * shade + raw_tex[:, :, 1]
            skin[:, :, 2] = cur_r * shade + raw_tex[:, :, 2]
            frame[face_mask] = np.clip(skin[face_mask], 0, 255).astype(np.uint8)

        else:
            if fake_type == "diffusion":
                # GENERATIVE AI (Sora / Kling / Runway / Midjourney):
                # Smooth 3D animation, but OVER-SMOOTHED skin (latent diffusion blur, no pore micro-texture)
                # ZERO physiological cardiac pulse (flatline or diffuse neural drift)
                drift = 1.0 * np.sin(2 * np.pi * 0.08 * t)
                cur_r = base_r + drift * 0.2
                cur_g = base_g + drift * 0.1
                cur_b = base_b + drift * 0.3

                skin = np.zeros((height, width, 3), dtype=np.float32)
                skin[:, :, 0] = cur_b * shade
                skin[:, :, 1] = cur_g * shade
                skin[:, :, 2] = cur_r * shade
                frame[face_mask] = np.clip(skin[face_mask], 0, 255).astype(np.uint8)

            elif fake_type == "talking_head":
                # AI TALKING-HEAD (Wav2Lip / SadTalker):
                # Over-smoothed skin, mouth animated separately
                skin = np.zeros((height, width, 3), dtype=np.float32)
                skin[:, :, 0] = base_b * shade
                skin[:, :, 1] = base_g * shade
                skin[:, :, 2] = base_r * shade
                frame[face_mask] = np.clip(skin[face_mask], 0, 255).astype(np.uint8)

            else:
                # FACESWAP:
                # Phase desynchronization and visible boundary seam
                jitter = 5.5 * np.sin(2 * np.pi * 6.0 * t) + np.random.normal(0, 1.5)
                cur_r = base_r + jitter * 0.4
                cur_g = base_g - jitter * 0.7
                cur_b = base_b + jitter * 1.0

                skin = np.zeros((height, width, 3), dtype=np.float32)
                skin[:, :, 0] = cur_b * shade + raw_tex[:, :, 0] * 0.5
                skin[:, :, 1] = cur_g * shade + raw_tex[:, :, 1] * 0.5
                skin[:, :, 2] = cur_r * shade + raw_tex[:, :, 2] * 0.5
                frame[face_mask] = np.clip(skin[face_mask], 0, 255).astype(np.uint8)

        # Draw Facial Features (Eyes, Eyebrows, Mouth)
        eye_y = cy - int(ry * 0.20)
        left_eye_x = cx - int(rx * 0.45)
        right_eye_x = cx + int(rx * 0.45)

        # Eyeballs
        cv2.circle(frame, (left_eye_x, eye_y), 10, (245, 245, 245), -1)
        cv2.circle(frame, (right_eye_x, eye_y), 10, (245, 245, 245), -1)
        # Pupils
        cv2.circle(frame, (left_eye_x, eye_y), 4, (30, 20, 20), -1)
        cv2.circle(frame, (right_eye_x, eye_y), 4, (30, 20, 20), -1)

        # Eyebrows
        cv2.ellipse(frame, (left_eye_x, eye_y - 12), (14, 4), 0, 0, 360, (50, 40, 30), -1)
        cv2.ellipse(frame, (right_eye_x, eye_y - 12), (14, 4), 0, 0, 360, (50, 40, 30), -1)

        # Mouth
        mouth_y = cy + int(ry * 0.50)
        mouth_open = int(3 + 2.5 * np.sin(2 * np.pi * 1.2 * t))
        cv2.ellipse(frame, (cx, mouth_y), (22, max(2, mouth_open)), 0, 0, 360, (60, 60, 180), -1)

        # Artifact Injection for Specific Deepfakes
        if is_fake:
            if fake_type == "faceswap":
                # Blending seam along outer jawline (characteristic FaceSwap border discontinuity)
                cv2.ellipse(frame, (cx, cy), (int(rx * 0.90), int(ry * 0.92)), 0, 0, 360, (35, 45, 35), 4)
            elif fake_type == "talking_head":
                # Lower face chrominance seam
                patch_y1, patch_y2 = max(0, mouth_y - 25), min(height, mouth_y + 35)
                patch_x1, patch_x2 = max(0, cx - 45), min(width, cx + 45)
                mouth_patch = frame[patch_y1:patch_y2, patch_x1:patch_x2].astype(np.float32)
                mouth_patch[:, :, 0] = np.clip(mouth_patch[:, :, 0] * 0.85, 0, 255)
                mouth_patch[:, :, 2] = np.clip(mouth_patch[:, :, 2] * 1.15, 0, 255)
                frame[patch_y1:patch_y2, patch_x1:patch_x2] = mouth_patch.astype(np.uint8)

        out.write(frame)

    out.release()
    return output_path
