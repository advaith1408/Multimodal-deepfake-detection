@echo off
echo ========================================================================
echo   Launching VeritasFace: Explainable Multimodal Deepfake Detector
echo   Architecture: EfficientNet-B0 + Bi-LSTM Attention + POS rPPG Fusion
echo ========================================================================
echo.
echo Starting FastAPI Web Server at http://127.0.0.1:8000 ...
echo Opening your browser to http://127.0.0.1:8000 ...
start http://127.0.0.1:8000
python run_server.py
pause
