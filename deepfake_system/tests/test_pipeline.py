"""
End-to-End Pipeline Verification Test
Generates synthetic benchmark videos and executes the complete forensic detection & reporting flow.
"""

import os
import unittest
from pathlib import Path

from deepfake_system.data.sample_generator import generate_benchmark_video
from deepfake_system.models.deepfake_detector import MultimodalDeepfakeDetector
from deepfake_system.explainability.gradcam import GradCAMExplainer
from deepfake_system.explainability.rppg_explainer import PhysiologicalExplainer
from deepfake_system.explainability.report_generator import ForensicReportGenerator


class TestEndToEndPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).resolve().parent / "sample_data"
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        cls.real_video = str(cls.test_dir / "test_real.mp4")
        cls.fake_video = str(cls.test_dir / "test_fake.mp4")

        print("Generating synthetic authentic benchmark video...")
        generate_benchmark_video(cls.real_video, is_fake=False, duration_sec=2.5, fps=30)
        print("Generating synthetic manipulated benchmark video...")
        generate_benchmark_video(cls.fake_video, is_fake=True, duration_sec=2.5, fps=30)

        cls.detector = MultimodalDeepfakeDetector()
        cls.gradcam = GradCAMExplainer(cls.detector.cnn, cls.detector.cnn.target_conv_layer)

    def test_standard_mode_detection(self):
        print("\n--- Testing Standard Mode (CNN + LSTM) ---")
        res = self.detector.predict_video(self.real_video, mode="standard")
        self.assertIn("prediction", res)
        self.assertIn("confidence", res)
        self.assertEqual(res["mode_used"], "standard")
        print(f"Standard Mode Output: Verdict={res['prediction']} (Confidence: {res['confidence']:.2f})")

    def test_advanced_mode_detection_and_rppg(self):
        print("\n--- Testing Advanced Mode (CNN + LSTM + rPPG) ---")
        res_real = self.detector.predict_video(self.real_video, mode="advanced")
        self.assertIn("rppg_analysis", res_real)
        coherence_real = res_real["rppg_analysis"].get("mean_cross_region_coherence", 0.0)
        print(f"Real Video - Coherence: {coherence_real:.3f}, HR: {res_real['rppg_analysis'].get('heart_rate_bpm'):.1f} BPM")

        res_fake = self.detector.predict_video(self.fake_video, mode="advanced")
        coherence_fake = res_fake["rppg_analysis"].get("mean_cross_region_coherence", 0.0)
        print(f"Fake Video - Coherence: {coherence_fake:.3f}")

        # Real video must demonstrate higher cross-regional physiological coherence than manipulated video
        self.assertGreater(coherence_real, coherence_fake)

    def test_explainability_and_pdf_generation(self):
        print("\n--- Testing Explainability & PDF Report Generation ---")
        res = self.detector.predict_video(self.fake_video, mode="advanced")
        res["file_hash"] = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

        # Grad-CAM
        sample_crop = res["sample_face_crops"][0]
        tensor = self.detector.preprocess_frame(sample_crop)
        cam_mask = self.gradcam.generate_cam(tensor, target_class=1)
        blended, _ = GradCAMExplainer.overlay_heatmap(sample_crop, cam_mask)
        gradcam_path = str(self.test_dir / "gradcam_output.png")
        import cv2
        cv2.imwrite(gradcam_path, blended)

        # PDF Report
        pdf_path = str(self.test_dir / "test_forensic_report.pdf")
        ForensicReportGenerator.generate_pdf_report(
            analysis_results=res,
            output_path=pdf_path,
            gradcam_img_path=gradcam_path
        )
        self.assertTrue(os.path.exists(pdf_path))
        print(f"Successfully generated forensic PDF report: {pdf_path} (Size: {os.path.getsize(pdf_path)} bytes)")


if __name__ == "__main__":
    unittest.main()
