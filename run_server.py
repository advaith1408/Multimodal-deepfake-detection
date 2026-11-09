"""
Production Server Launcher
Explainable Multimodal Deepfake Face Detection System
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Starting VeritasFace Server on http://{host}:{port} ...")
    uvicorn.run("deepfake_system.api.app:app", host=host, port=port, reload=False)
