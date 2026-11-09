"""
FastAPI Endpoints Verification Script (Synchronous Direct Execution)
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepfake_system.api.app import (
    health_check, serve_index, get_benchmark_results, analyze_video, generate_pdf
)
from fastapi import UploadFile

print("1. Testing health_check()...", flush=True)
res_health = health_check()
print("   Health:", res_health, flush=True)
assert res_health["status"] == "healthy"

print("2. Testing serve_index()...", flush=True)
res_index = serve_index()
print("   Index File Path:", res_index.path, flush=True)
assert Path(res_index.path).exists()

print("3. Testing get_benchmark_results()...", flush=True)
res_bench = get_benchmark_results()
print("   Benchmark keys:", list(res_bench["metrics"].keys()), flush=True)
assert "cnn_lstm_rppg" in res_bench["metrics"]

print("4. Testing analyze_video() (Advanced Mode with rPPG)...", flush=True)
sample_path = str(Path(__file__).resolve().parent / "deepfake_system/api/static/samples/sample_real.mp4")
with open(sample_path, "rb") as f:
    upload_file = UploadFile(filename="sample_real.mp4", file=f)
    pred_response = analyze_video(video=upload_file, mode="advanced")
    pred_data = json.loads(pred_response.body)
    print("   Prediction:", pred_data["prediction"], "Confidence:", pred_data["confidence"], flush=True)
    print("   rPPG Coherence:", pred_data["rppg_summary"]["cross_region_coherence"], flush=True)
    assert pred_data["prediction"] in ["REAL", "FAKE"]
    assert pred_data["gradcam_image"] is not None

print("5. Testing analyze_video() (Standard Mode)...", flush=True)
with open(sample_path, "rb") as f:
    upload_file = UploadFile(filename="sample_real.mp4", file=f)
    pred_response = analyze_video(video=upload_file, mode="standard")
    pred_data_std = json.loads(pred_response.body)
    print("   Prediction:", pred_data_std["prediction"], "Mode:", pred_data_std["mode_used"], flush=True)
    assert pred_data_std["mode_used"] == "standard"

print("6. Testing generate_pdf()...", flush=True)
pdf_response = generate_pdf(pred_data)
print("   PDF Path:", pdf_response.path, "Size:", Path(pdf_response.path).stat().st_size, "bytes", flush=True)
assert Path(pdf_response.path).exists()
assert Path(pdf_response.path).stat().st_size > 5000

print("\n=======================================================")
print("ALL 6 FASTAPI ENDPOINTS TESTED AND VERIFIED PASSING!")
print("=======================================================\n", flush=True)
