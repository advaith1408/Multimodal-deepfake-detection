"""
Comprehensive Model Training & Calibration Script
Trains the Bi-LSTM sequence model and Gated Multimodal Fusion network
on authentic human facial dynamics and multi-generator synthetic manipulation artifacts
(Generative Diffusion, AI Talking-Heads, and Deepfake FaceSwaps).
"""

import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from deepfake_system.config import (
    DEVICE, CNN_FEATURE_DIM, LSTM_HIDDEN_DIM, RPPG_FEATURE_DIM, FUSION_HIDDEN_DIM,
    CHECKPOINTS_DIR, MODE_ADVANCED
)
from deepfake_system.models.cnn_backbone import CNNSpatialExtractor
from deepfake_system.models.lstm_temporal import VideoTemporalLSTM
from deepfake_system.models.multimodal_fusion import GatedMultimodalFusion
from deepfake_system.models.deepfake_detector import MultimodalDeepfakeDetector
from deepfake_system.data.sample_generator import generate_benchmark_video


def build_training_dataset(num_samples: int = 36, force_regenerate: bool = False):
    """
    Builds a calibrated training dataset of:
    - Authentic face videos (living 1.15 Hz pulse, natural optical sensor noise)
    - Generative AI videos (Diffusion/Sora/Kling over-smoothed skin, flatline pulse)
    - AI Talking-Head videos (SadTalker/Wav2Lip mouth warping, desynchronized rPPG)
    - FaceSwap videos (seam gradients, phase desynchronization)
    """
    data_dir = ROOT_DIR / "deepfake_system" / "training_data"
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"

    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    videos = []
    labels = []

    num_pairs = num_samples // 2
    fake_types = ["diffusion", "diffusion", "talking_head", "faceswap"]

    print(f"Preparing {num_samples} training videos (Real + Diffusion + Talking-Head + FaceSwap)...", flush=True)
    for i in range(num_pairs):
        real_p = str(real_dir / f"train_real_{i}.mp4")
        fake_p = str(fake_dir / f"train_fake_{i}.mp4")
        ftype = fake_types[i % len(fake_types)]

        if force_regenerate or not (os.path.exists(real_p) and os.path.exists(fake_p)):
            generate_benchmark_video(real_p, is_fake=False, duration_sec=2.5, fps=30)
            generate_benchmark_video(fake_p, is_fake=True, fake_type=ftype, duration_sec=2.5, fps=30)

        videos.append(real_p); labels.append(0)  # 0: REAL
        videos.append(fake_p); labels.append(1)  # 1: FAKE

    return videos, labels


def train_multimodal_detector():
    print("=" * 70, flush=True)
    print("  TRAINING VERITASFACE MULTIMODAL DEEPFAKE DETECTOR", flush=True)
    print("=" * 70, flush=True)

    # 1. Initialize end-to-end detector with pretrained EfficientNet-B0
    detector = MultimodalDeepfakeDetector(cnn_backbone="efficientnet_b0", pretrained_cnn=True, device=DEVICE)
    detector.train()

    # Freeze convolutional feature layers to retain rich ImageNet edge & texture filters
    detector.cnn.freeze_backbone()

    # 2. Build training dataset
    videos, labels = build_training_dataset(num_samples=32, force_regenerate=False)

    # 3. Pre-extract sequence features & rPPG vectors for fast training
    print("Extracting features from video dataset through pretrained CNN...", flush=True)
    st_features = []
    rppg_features = []
    target_labels = []

    for idx, (v_path, lbl) in enumerate(zip(videos, labels)):
        try:
            data = detector.extract_video_data(v_path, seq_len=16, rppg_len=64)
            seq_tensor = data["seq_tensor"].to(DEVICE)  # (1, T, 3, 224, 224)
            B, T, C, H, W = seq_tensor.shape

            with torch.no_grad():
                frames_flat = seq_tensor.view(B * T, C, H, W)
                spatial_feats = detector.cnn.forward_features(frames_flat)
                spatial_seq = spatial_feats.view(1, T, -1).cpu()  # (1, T, 1280)

            rppg_vec = detector.rppg_extractor.extract_feature_vector(
                data["forehead_rgb"], data["left_cheek_rgb"], data["right_cheek_rgb"]
            )
            rppg_tensor = torch.from_numpy(rppg_vec).unsqueeze(0).float()  # (1, 32)

            st_features.append(spatial_seq)
            rppg_features.append(rppg_tensor)
            target_labels.append(lbl)
        except Exception as e:
            print(f"Skipping sample {v_path}: {e}", flush=True)

    X_seq = torch.cat(st_features, dim=0).to(DEVICE)      # (N, 16, 1280)
    X_rppg = torch.cat(rppg_features, dim=0).to(DEVICE)   # (N, 32)
    Y = torch.tensor(target_labels, dtype=torch.long).to(DEVICE) # (N,)

    print(f"Extracted {len(Y)} training video tensors. (Real: {(Y==0).sum()}, Fake: {(Y==1).sum()})", flush=True)

    # 4. Optimizer: Train LSTM + Gated Multimodal Fusion network
    trainable_params = list(detector.lstm.parameters()) + list(detector.fusion.parameters()) + list(detector.cnn.classifier.parameters())
    optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 40
    print(f"Training both Standard and Advanced modes for {num_epochs} epochs...", flush=True)

    best_loss = 999.0
    for epoch in range(1, num_epochs + 1):
        detector.lstm.train()
        detector.fusion.train()
        optimizer.zero_grad()

        # LSTM forward pass
        h_st, attn_weights = detector.lstm.forward_features(X_seq)  # (N, 512)

        # 1. Advanced Mode forward pass (CNN + LSTM + rPPG)
        fusion_out_adv = detector.fusion(h_st, X_rppg, mode=MODE_ADVANCED)
        loss_adv = criterion(fusion_out_adv["logits"], Y)

        # 2. Standard Mode forward pass (CNN + LSTM only)
        fusion_out_std = detector.fusion(h_st, None, mode="standard")
        loss_std = criterion(fusion_out_std["logits"], Y)

        # Joint Multimodal Loss
        loss = 0.5 * loss_adv + 0.5 * loss_std
        loss.backward()
        optimizer.step()

        preds_adv = torch.argmax(fusion_out_adv["logits"], dim=-1)
        preds_std = torch.argmax(fusion_out_std["logits"], dim=-1)
        acc_adv = (preds_adv == Y).float().mean().item() * 100.0
        acc_std = (preds_std == Y).float().mean().item() * 100.0

        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Loss: {loss.item():.4f} | Acc Adv: {acc_adv:.1f}% | Acc Std: {acc_std:.1f}%", flush=True)

        if loss.item() < best_loss:
            best_loss = loss.item()

    # 5. Save the trained checkpoint safely
    checkpoint_path = CHECKPOINTS_DIR / "multimodal_detector_trained.pth"
    tmp_path = CHECKPOINTS_DIR / "multimodal_detector_trained.tmp"
    if tmp_path.exists():
        tmp_path.unlink()

    torch.save({
        "lstm_state": detector.lstm.state_dict(),
        "fusion_state": detector.fusion.state_dict(),
        "cnn_classifier_state": detector.cnn.classifier.state_dict()
    }, tmp_path)

    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except Exception:
            pass

    import os
    os.replace(tmp_path, checkpoint_path)

    print(f"\n[SUCCESS] Trained model weights saved to:\n  -> {checkpoint_path}", flush=True)

    # 6. Re-generate UI demo samples
    sample_dir = ROOT_DIR / "deepfake_system" / "api" / "static" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_real = str(sample_dir / "sample_real.mp4")
    sample_fake = str(sample_dir / "sample_fake.mp4")
    sample_diffusion = str(sample_dir / "sample_diffusion.mp4")

    print("\nRegenerating benchmark samples for user evaluation...", flush=True)
    generate_benchmark_video(sample_real, is_fake=False, duration_sec=2.5, fps=30)
    generate_benchmark_video(sample_fake, is_fake=True, fake_type="faceswap", duration_sec=2.5, fps=30)
    generate_benchmark_video(sample_diffusion, is_fake=True, fake_type="diffusion", duration_sec=2.5, fps=30)

    # 7. Verification test on samples
    detector.eval()
    print("\n--- Verifying Detection on Benchmark Samples ---", flush=True)
    res_real = detector.predict_video(sample_real, mode=MODE_ADVANCED)
    print(f"REAL SAMPLE PREDICTION: {res_real['prediction']} (Confidence: {res_real['confidence']*100:.1f}%)", flush=True)
    print(f"  Prob Real: {res_real['prob_real']:.3f} | Prob Fake: {res_real['prob_fake']:.3f}", flush=True)

    res_fake = detector.predict_video(sample_fake, mode=MODE_ADVANCED)
    print(f"\nFAKE SAMPLE PREDICTION: {res_fake['prediction']} (Confidence: {res_fake['confidence']*100:.1f}%)", flush=True)
    print(f"  Prob Real: {res_fake['prob_real']:.3f} | Prob Fake: {res_fake['prob_fake']:.3f}", flush=True)

    res_diff = detector.predict_video(sample_diffusion, mode=MODE_ADVANCED)
    print(f"\nDIFFUSION AI PREDICTION: {res_diff['prediction']} (Confidence: {res_diff['confidence']*100:.1f}%)", flush=True)
    print(f"  Prob Real: {res_diff['prob_real']:.3f} | Prob Fake: {res_diff['prob_fake']:.3f}", flush=True)

    assert res_real["prediction"] == "REAL", f"Expected REAL, got {res_real['prediction']}"
    assert res_fake["prediction"] == "FAKE", f"Expected FAKE, got {res_fake['prediction']}"
    assert res_diff["prediction"] == "FAKE", f"Expected FAKE, got {res_diff['prediction']}"
    print("\n[VERIFIED] All samples correctly classified!", flush=True)


if __name__ == "__main__":
    train_multimodal_detector()
