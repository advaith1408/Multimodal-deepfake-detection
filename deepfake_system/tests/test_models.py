"""
Unit Tests for Deepfake Detection Components
Tests Face & ROI Detector, rPPG Extractor, CNN Spatial, LSTM Temporal, and Gated Fusion.
"""

import unittest
import numpy as np
import torch

from deepfake_system.data.face_detector import FaceAndROIDetector
from deepfake_system.models.rppg_extractor import RPPGExtractor
from deepfake_system.models.cnn_backbone import CNNSpatialExtractor
from deepfake_system.models.lstm_temporal import VideoTemporalLSTM
from deepfake_system.models.multimodal_fusion import GatedMultimodalFusion
from deepfake_system.explainability.gradcam import GradCAMExplainer


class TestDeepfakeSystem(unittest.TestCase):

    def test_face_and_roi_detector(self):
        detector = FaceAndROIDetector()
        # Create synthetic face-like RGB frame
        frame = np.full((320, 320, 3), 40, dtype=np.uint8)
        cv2_bgr_face = (140, 168, 215)
        import cv2
        cv2.ellipse(frame, (160, 160), (90, 115), 0, 0, 360, cv2_bgr_face, -1)

        bbox = detector.detect_face(frame)
        self.assertIsNotNone(bbox)
        self.assertEqual(len(bbox), 4)

        crop = detector.crop_face(frame, (224, 224))
        self.assertEqual(crop.shape, (224, 224, 3))

        rois = detector.extract_rois(frame)
        self.assertIn("forehead", rois)
        self.assertIn("left_cheek", rois)
        self.assertIn("right_cheek", rois)

        rgb_mean = detector.compute_roi_mean_rgb(rois["forehead"])
        self.assertEqual(len(rgb_mean), 3)

    def test_rppg_extractor(self):
        extractor = RPPGExtractor(fps=30.0, method="POS")
        # Simulate 60 frames with 1.2 Hz pulse
        t = np.linspace(0, 2.0, 60)
        pulse = 4.0 * np.sin(2 * np.pi * 1.2 * t)
        rgb_series = np.column_stack([
            210 + pulse * 0.2,
            165 + pulse * 1.0,
            140 + pulse * 0.1
        ])

        bvp = extractor.extract_bvp(rgb_series)
        self.assertEqual(len(bvp), 60)

        spec = extractor.analyze_spectrum(bvp)
        self.assertIn("heart_rate_bpm", spec)
        self.assertIn("snr_db", spec)
        self.assertGreater(spec["heart_rate_bpm"], 40.0)

        # Multi-region coherence
        fh_rgb = rgb_series
        lc_rgb = rgb_series + np.random.normal(0, 0.2, rgb_series.shape)
        rc_rgb = rgb_series + np.random.normal(0, 0.2, rgb_series.shape)
        coherence = extractor.compute_multi_region_coherence(fh_rgb, lc_rgb, rc_rgb)
        self.assertIn("mean_cross_region_coherence", coherence)
        self.assertGreater(coherence["mean_cross_region_coherence"], 0.7)

        feat_vec = extractor.extract_feature_vector(fh_rgb, lc_rgb, rc_rgb)
        self.assertEqual(len(feat_vec), 32)

    def test_cnn_spatial_extractor(self):
        cnn = CNNSpatialExtractor(backbone_name="efficientnet_b0", pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224)
        
        # Test feature extraction
        feats = cnn.forward_features(dummy_input)
        self.assertEqual(feats.shape, (2, 1280))

        # Test classification logits
        logits = cnn(dummy_input)
        self.assertEqual(logits.shape, (2, 2))

    def test_lstm_temporal(self):
        lstm = VideoTemporalLSTM(input_dim=1280, hidden_dim=256)
        dummy_seq = torch.randn(2, 16, 1280)  # Batch=2, T=16, Dim=1280

        h_st, attn = lstm.forward_features(dummy_seq)
        self.assertEqual(h_st.shape, (2, 512))
        self.assertEqual(attn.shape, (2, 16))
        # Verify attention weights sum to 1.0 across timesteps
        self.assertTrue(torch.allclose(torch.sum(attn, dim=1), torch.ones(2), atol=1e-4))

    def test_gated_multimodal_fusion(self):
        fusion = GatedMultimodalFusion(st_dim=512, rppg_dim=32, fusion_dim=256)
        h_st = torch.randn(2, 512)
        h_rppg = torch.randn(2, 32)

        # Mode: Standard
        res_std = fusion(h_st, h_rppg, mode="standard")
        self.assertEqual(res_std["logits"].shape, (2, 2))
        self.assertEqual(res_std["probabilities"].shape, (2, 2))

        # Mode: Advanced
        res_adv = fusion(h_st, h_rppg, mode="advanced")
        self.assertEqual(res_adv["logits"].shape, (2, 2))
        self.assertEqual(res_adv["probabilities"].shape, (2, 2))
        self.assertEqual(res_adv["gate_weight"].shape, (2, 1))

    def test_gradcam(self):
        cnn = CNNSpatialExtractor(backbone_name="efficientnet_b0", pretrained=False)
        explainer = GradCAMExplainer(cnn, cnn.target_conv_layer)
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

        cam = explainer.generate_cam(dummy_input, target_class=1)
        self.assertEqual(cam.shape, (224, 224))
        self.assertTrue(np.all(cam >= 0.0) and np.all(cam <= 1.0))


if __name__ == "__main__":
    unittest.main()
