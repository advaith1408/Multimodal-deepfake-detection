import os
import sys
sys.path.insert(0, r"C:\major project")
from deepfake_system.models.deepfake_detector import MultimodalDeepfakeDetector, MODE_ADVANCED, MODE_STANDARD

detector = MultimodalDeepfakeDetector()

videos = {
    "WIN_webcam": r"C:/Users/advai/.gemini/antigravity/brain/1a737500-8c8c-4a65-a52e-5424079a9202/.user_uploaded/media_1794215352354_9aa94ef8.mp4",
    "karthik": r"C:/Users/advai/.gemini/antigravity/brain/1a737500-8c8c-4a65-a52e-5424079a9202/.user_uploaded/media_1794220170182_3650bc28.mp4",
    "harin": r"C:/Users/advai/.gemini/antigravity/brain/1a737500-8c8c-4a65-a52e-5424079a9202/.user_uploaded/media_1794220179165_5e0ecdf9.mp4",
    "fake_swap": r"C:/major project/deepfake_system/api/static/samples/sample_fake.mp4",
    "diffusion": r"C:/major project/deepfake_system/api/static/samples/sample_diffusion.mp4"
}

for name, path in videos.items():
    res_adv = detector.predict_video(path, mode=MODE_ADVANCED)
    res_std = detector.predict_video(path, mode=MODE_STANDARD)
    print(f"=== {name} ===")
    print(f"  ADV: {res_adv['prediction']} (conf: {res_adv['confidence']:.3f}, p_real: {res_adv['prob_real']:.3f}, p_fake: {res_adv['prob_fake']:.3f})")
    print(f"  STD: {res_std['prediction']} (conf: {res_std['confidence']:.3f}, p_real: {res_std['prob_real']:.3f}, p_fake: {res_std['prob_fake']:.3f})")
    rppg = res_adv['rppg_analysis']
    spat = res_adv['spatial_forensics']
    print(f"  rPPG: HR={rppg.get('heart_rate_bpm', 0):.1f}, coh={rppg.get('mean_cross_region_coherence', 0):.3f}, ppr={rppg.get('peak_power_ratio', 0):.3f}, ent={rppg.get('spectral_entropy', 0):.3f}, snr={rppg.get('snr_db', 0):.1f}")
    print(f"  Spat: lap={spat.get('mean_laplacian_var', 0):.1f}, hf={spat.get('mean_hf_ratio', 0):.3f}, bgrad={spat.get('mean_boundary_grad', 0):.1f}, fake_score={spat.get('spatial_fake_score', 0):.3f}")
