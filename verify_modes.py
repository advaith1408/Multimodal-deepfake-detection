import sys
from pathlib import Path
sys.path.insert(0, 'C:/major project')

from deepfake_system.models.deepfake_detector import MultimodalDeepfakeDetector

detector = MultimodalDeepfakeDetector(pretrained_cnn=True)
real_p = 'C:/major project/deepfake_system/api/static/samples/sample_real.mp4'
fake_p = 'C:/major project/deepfake_system/api/static/samples/sample_fake.mp4'
diff_p = 'C:/major project/deepfake_system/api/static/samples/sample_diffusion.mp4'

res_real_adv = detector.predict_video(real_p, mode='advanced')
res_real_std = detector.predict_video(real_p, mode='standard')

res_fake_adv = detector.predict_video(fake_p, mode='advanced')
res_fake_std = detector.predict_video(fake_p, mode='standard')

res_diff_adv = detector.predict_video(diff_p, mode='advanced')
res_diff_std = detector.predict_video(diff_p, mode='standard')

print("=" * 70)
print(f"REAL Video (Advanced Mode)   : {res_real_adv['prediction']} (Confidence: {res_real_adv['confidence']*100:.1f}%) | Prob Real: {res_real_adv['prob_real']:.3f}")
print(f"REAL Video (Standard Mode)   : {res_real_std['prediction']} (Confidence: {res_real_std['confidence']*100:.1f}%) | Prob Real: {res_real_std['prob_real']:.3f}")
print(f"FACESWAP Video (Advanced)    : {res_fake_adv['prediction']} (Confidence: {res_fake_adv['confidence']*100:.1f}%) | Prob Fake: {res_fake_adv['prob_fake']:.3f}")
print(f"FACESWAP Video (Standard)    : {res_fake_std['prediction']} (Confidence: {res_fake_std['confidence']*100:.1f}%) | Prob Fake: {res_fake_std['prob_fake']:.3f}")
print(f"DIFFUSION AI Video (Advanced): {res_diff_adv['prediction']} (Confidence: {res_diff_adv['confidence']*100:.1f}%) | Prob Fake: {res_diff_adv['prob_fake']:.3f}")
print(f"DIFFUSION AI Video (Standard): {res_diff_std['prediction']} (Confidence: {res_diff_std['confidence']*100:.1f}%) | Prob Fake: {res_diff_std['prob_fake']:.3f}")
print("=" * 70)

assert res_real_adv['prediction'] == 'REAL', f"Real video failed in Advanced Mode: {res_real_adv['prediction']}"
assert res_real_std['prediction'] == 'REAL', f"Real video failed in Standard Mode: {res_real_std['prediction']}"
assert res_fake_adv['prediction'] == 'FAKE', f"FaceSwap video failed in Advanced Mode: {res_fake_adv['prediction']}"
assert res_fake_std['prediction'] == 'FAKE', f"FaceSwap video failed in Standard Mode: {res_fake_std['prediction']}"
assert res_diff_adv['prediction'] == 'FAKE', f"Diffusion video failed in Advanced Mode: {res_diff_adv['prediction']}"
assert res_diff_std['prediction'] == 'FAKE', f"Diffusion video failed in Standard Mode: {res_diff_std['prediction']}"

print("[VERIFIED] Complete bi-directional accuracy: Real videos classified as REAL, AI/Deepfake videos classified as FAKE across all modes!")
