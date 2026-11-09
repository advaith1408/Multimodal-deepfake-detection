import sys
import os
sys.path.insert(0, r"C:\major project")
from deepfake_system.models.deepfake_detector import MultimodalDeepfakeDetector

d = MultimodalDeepfakeDetector()
d.eval()

test_vids = {
    "Real: Webcam": r"C:\Users\advai\.gemini\antigravity\brain\1a737500-8c8c-4a65-a52e-5424079a9202\.user_uploaded\media_1794215352354_9aa94ef8.mp4",
    "Real: Karthik": r"C:\Users\advai\.gemini\antigravity\brain\1a737500-8c8c-4a65-a52e-5424079a9202\.user_uploaded\media_1794220170182_3650bc28.mp4",
    "Real: Harin": r"C:\Users\advai\.gemini\antigravity\brain\1a737500-8c8c-4a65-a52e-5424079a9202\.user_uploaded\media_1794220179165_5e0ecdf9.mp4",
    "Real: Campus": r"C:\Users\advai\Downloads\kollijpvedio.mp4",
    "Fake: Sample FaceSwap": r"C:\major project\deepfake_system\api\static\samples\sample_fake.mp4",
    "Fake: Gemini 1": r"C:\Users\advai\Downloads\gemini_generated_video_08e3eae4.mp4",
    "Fake: Gemini 2": r"C:\Users\advai\Downloads\gemini_generated_video_d0766dae.mp4",
    "Fake: Gemini 3": r"C:\Users\advai\Downloads\gemini_generated_video_255e3a82.mp4"
}

print(f"{'Video Test Case':<22} | {'Advanced Mode':<20} | {'Standard Mode':<20} | {'Key Forensic Metrics'}")
print("-" * 90)

for name, p in test_vids.items():
    if not os.path.exists(p):
        continue
    res_adv = d.predict_video(p, mode="advanced")
    res_std = d.predict_video(p, mode="standard")
    
    pred_adv = res_adv["prediction"]
    conf_adv = res_adv["confidence"] * 100
    
    pred_std = res_std["prediction"]
    conf_std = res_std["confidence"] * 100

    spat = res_adv["spatial_forensics"]
    rppg = res_adv.get("rppg_analysis", {})
    lap = spat.get("mean_laplacian_var", 0.0)
    bg = spat.get("mean_boundary_grad", 0.0)
    coh = rppg.get("mean_cross_region_coherence", 0.0)

    print(f"{name:<22} | {pred_adv:<4} ({conf_adv:5.1f}% conf)     | {pred_std:<4} ({conf_std:5.1f}% conf)     | LapVar={lap:5.1f}, BGrad={bg:4.1f}, Coh={coh:.3f}")
