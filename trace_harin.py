import sys, torch
sys.path.insert(0, 'C:/major project')
from deepfake_system.models.deepfake_detector import MultimodalDeepfakeDetector

v_path = r'C:/Users/advai/.gemini/antigravity/brain/1a737500-8c8c-4a65-a52e-5424079a9202/.user_uploaded/media_1794220179165_5e0ecdf9.mp4'
detector = MultimodalDeepfakeDetector(pretrained_cnn=True)
detector.eval()

data = detector.extract_video_data(v_path)
seq_tensor = data["seq_tensor"]
B, T, C, H, W = seq_tensor.shape

with torch.no_grad():
    frames_flat = seq_tensor.view(B * T, C, H, W)
    spatial_feats = detector.cnn.forward_features(frames_flat)
    spatial_seq = spatial_feats.view(B, T, -1)
    h_st, attn_weights = detector.lstm.forward_features(spatial_seq)
    
    # 1. Standard mode raw fusion:
    fusion_std = detector.fusion(h_st, None, mode="standard")
    probs_std = fusion_std["probabilities"][0].cpu().numpy()
    print("probs_std from fusion:", probs_std)
    
    # 2. Advanced mode raw fusion:
    rppg_feat_vec = detector.rppg_extractor.extract_feature_vector(
        data["forehead_rgb"], data["left_cheek_rgb"], data["right_cheek_rgb"]
    )
    h_rppg_tensor = torch.from_numpy(rppg_feat_vec).unsqueeze(0).to(detector.device)
    fusion_adv = detector.fusion(h_st, h_rppg_tensor, mode="advanced")
    probs_adv = fusion_adv["probabilities"][0].cpu().numpy()
    print("probs_adv from fusion:", probs_adv)

rppg_metrics = detector.rppg_extractor.compute_multi_region_coherence(
    data["forehead_rgb"], data["left_cheek_rgb"], data["right_cheek_rgb"]
)
print("rppg_metrics scalar keys:")
for k in ['mean_cross_region_coherence', 'heart_rate_bpm', 'snr_db', 'spectral_entropy', 'peak_power_ratio']:
    print(f"  {k}: {rppg_metrics.get(k)}")
print("fps:", data["fps"])
