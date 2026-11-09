import cv2, sys, torch
import numpy as np
sys.path.insert(0, 'C:/major project')
from deepfake_system.models.deepfake_detector import MultimodalDeepfakeDetector

detector = MultimodalDeepfakeDetector(pretrained_cnn=True)
detector.eval()

videos = [
    ("karthik (real)", r"C:/Users/advai/.gemini/antigravity/brain/1a737500-8c8c-4a65-a52e-5424079a9202/.user_uploaded/media_1794220170182_3650bc28.mp4"),
    ("harin (real)", r"C:/Users/advai/.gemini/antigravity/brain/1a737500-8c8c-4a65-a52e-5424079a9202/.user_uploaded/media_1794220179165_5e0ecdf9.mp4"),
    ("win_user (real)", r"C:/Users/advai/.gemini/antigravity/brain/1a737500-8c8c-4a65-a52e-5424079a9202/.user_uploaded/media_1794215352354_9aa94ef8.mp4"),
    ("sample_fake", "C:/major project/deepfake_system/api/static/samples/sample_fake.mp4"),
    ("sample_diff", "C:/major project/deepfake_system/api/static/samples/sample_diffusion.mp4")
]

for name, p in videos:
    data = detector.extract_video_data(p)
    seq = data["seq_tensor"]
    B, T, C, H, W = seq.shape
    with torch.no_grad():
        feats = detector.cnn.forward_features(seq.view(B*T, C, H, W)) # (T, 1280)
        # Inter-frame feature variance (temporal jitter)
        feat_diffs = torch.norm(feats[1:] - feats[:-1], dim=-1)
        mean_diff = feat_diffs.mean().item()
        std_diff = feat_diffs.std().item()
        jitter_ratio = std_diff / (mean_diff + 1e-6)
        
    spatial = detector.compute_spatial_forensics(data['face_crops'])
    rppg = detector.rppg_extractor.compute_multi_region_coherence(
        data["forehead_rgb"], data["left_cheek_rgb"], data["right_cheek_rgb"]
    )
    print(f"\n=== {name} ===")
    print(f"  Temporal Feature Drift: mean={mean_diff:.2f}, std={std_diff:.2f}, jitter={jitter_ratio:.2f}")
    print(f"  Spatial: mean_lap={spatial['mean_laplacian_var']:.1f}, mean_hf={spatial['mean_hf_ratio']:.3f}, bgrad={spatial['mean_boundary_grad']:.1f}")
    print(f"  rPPG: coherence={rppg.get('mean_cross_region_coherence',0):.3f}, HR={rppg.get('heart_rate_bpm',0):.1f}, entropy={rppg.get('spectral_entropy',1):.3f}")
