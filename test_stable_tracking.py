import cv2, sys, torch
import numpy as np
sys.path.insert(0, 'C:/major project')
from deepfake_system.models.deepfake_detector import MultimodalDeepfakeDetector
from deepfake_system.config import FRAME_SIZE

v_path = r"C:/Users/advai/.gemini/antigravity/brain/1a737500-8c8c-4a65-a52e-5424079a9202/.user_uploaded/media_1794220170182_3650bc28.mp4"
detector = MultimodalDeepfakeDetector(pretrained_cnn=True)
detector.eval()

cap = cv2.VideoCapture(v_path)
frames = []
while True:
    ret, frame = cap.read()
    if not ret: break
    frames.append(frame)
cap.release()

# 1. Detect all face candidates to find true face location
detected_boxes = []
for f in frames:
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    faces = detector.face_detector.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if len(faces) > 0:
        largest = max(faces, key=lambda b: b[2] * b[3])
        detected_boxes.append(largest)

print(f"Detected {len(detected_boxes)} face detections across {len(frames)} frames")
# Cluster/median of true face detections
med_x = int(np.median([b[0] for b in detected_boxes]))
med_y = int(np.median([b[1] for b in detected_boxes]))
med_w = int(np.median([b[2] for b in detected_boxes]))
med_h = int(np.median([b[3] for b in detected_boxes]))
print(f"Median face location: x={med_x}, y={med_y}, w={med_w}, h={med_h}")

# Now sample 16 frames using robust outlier rejection
step = max(1, len(frames) // 16)
indices = [min(i * step, len(frames)-1) for i in range(16)]

crops = []
tensors = []
for idx in indices:
    f = frames[idx]
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    faces = detector.face_detector.face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(50, 50))
    chosen_box = None
    if len(faces) > 0:
        # Choose face closest to median face location
        closest = min(faces, key=lambda b: (b[0]-med_x)**2 + (b[1]-med_y)**2)
        dist = np.sqrt((closest[0]-med_x)**2 + (closest[1]-med_y)**2)
        if dist < med_w * 0.8: # Reject outlier (e.g. firecracker at bottom of screen)
            chosen_box = closest
    
    if chosen_box is None:
        chosen_box = (med_x, med_y, med_w, med_h)
    
    x, y, w, h = chosen_box
    pad_x, pad_y = int(0.10 * w), int(0.10 * h)
    H, W = f.shape[:2]
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(W, x + w + pad_x), min(H, y + h + pad_y)
    crop = f[y1:y2, x1:x2]
    resized_crop = cv2.resize(crop, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    crops.append(resized_crop)
    tensors.append(detector.preprocess_frame(resized_crop))

seq_tensor = torch.cat(tensors, dim=0).unsqueeze(0)
B, T, C, H, W = seq_tensor.shape

with torch.no_grad():
    feats = detector.cnn.forward_features(seq_tensor.view(B*T, C, H, W)).view(B, T, -1)
    diffs = torch.norm(feats[0, 1:] - feats[0, :-1], dim=-1)
    print(f"Stable Temporal Drift: mean={diffs.mean().item():.2f}, std={diffs.std().item():.2f}, max={diffs.max().item():.2f}")
    h_st, attn = detector.lstm.forward_features(feats)
    std_out = detector.fusion(h_st, None, mode="standard")
    print(f"STABLE Raw CNN-LSTM Logits: {std_out['logits'][0].tolist()}")
    print(f"STABLE Raw CNN-LSTM Probs: real={std_out['probabilities'][0][0]:.4f}, fake={std_out['probabilities'][0][1]:.4f}")
