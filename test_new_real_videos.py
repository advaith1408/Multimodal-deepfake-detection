import sys, os
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, 'C:/major project')
from deepfake_system.models.deepfake_detector import MultimodalDeepfakeDetector

v1 = r"C:/Users/advai/.gemini/antigravity/brain/1a737500-8c8c-4a65-a52e-5424079a9202/.user_uploaded/media_1794220170182_3650bc28.mp4"
v2 = r"C:/Users/advai/.gemini/antigravity/brain/1a737500-8c8c-4a65-a52e-5424079a9202/.user_uploaded/media_1794220179165_5e0ecdf9.mp4"

videos = [("karthik_cracker", v1), ("harin_laughing", v2)]

detector = MultimodalDeepfakeDetector(pretrained_cnn=True)
detector.eval()

for name, v_path in videos:
    print(f"\n{'='*30} {name} {'='*30}")
    print(f"Path exists: {os.path.exists(v_path)}")
    cap = cv2.VideoCapture(v_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Specs: {width}x{height}, {total_frames} frames @ {fps:.1f} fps")

    try:
        data = detector.extract_video_data(v_path)
        print(f"Total face crops extracted: {len(data['face_crops'])}")
        
        # Test neural network raw outputs
        seq = data["seq_tensor"]
        B, T, C, H, W = seq.shape
        import torch
        with torch.no_grad():
            feats = detector.cnn.forward_features(seq.view(B*T, C, H, W)).view(B, T, -1)
            h_st, attn = detector.lstm.forward_features(feats)
            std_out = detector.fusion(h_st, None, mode="standard")
            print(f"Raw CNN-LSTM Probs: real={std_out['probabilities'][0][0]:.4f}, fake={std_out['probabilities'][0][1]:.4f}")
            print(f"Raw CNN-LSTM Logits: {std_out['logits'][0].tolist()}")

        spatial = detector.compute_spatial_forensics(data['face_crops'])
        print(f"Spatial Forensics: {spatial}")

        rppg = detector.rppg_extractor.compute_multi_region_coherence(
            data["forehead_rgb"], data["left_cheek_rgb"], data["right_cheek_rgb"]
        )
        print(f"rPPG Coherence: {rppg.get('mean_cross_region_coherence', 0):.4f}, HR: {rppg.get('heart_rate_bpm', 0):.1f}, Entropy: {rppg.get('spectral_entropy', 1):.4f}")

        res_adv = detector.predict_video(v_path, mode="advanced")
        res_std = detector.predict_video(v_path, mode="standard")
        print(f"Prediction ADV: {res_adv['prediction']} (conf: {res_adv['confidence']*100:.1f}%, prob_real: {res_adv['prob_real']:.4f}, prob_fake: {res_adv['prob_fake']:.4f})")
        print(f"Prediction STD: {res_std['prediction']} (conf: {res_std['confidence']*100:.1f}%, prob_real: {res_std['prob_real']:.4f}, prob_fake: {res_std['prob_fake']:.4f})")

    except Exception as e:
        import traceback
        print(f"Error on {name}: {e}")
        traceback.print_exc()
