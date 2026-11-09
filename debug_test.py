import sys
print("Step 1: start", flush=True)
from deepfake_system.api.app import app
print("Step 2: imported app", flush=True)
from fastapi.testclient import TestClient
print("Step 3: imported TestClient", flush=True)
client = TestClient(app)
print("Step 4: created client", flush=True)
r = client.get("/api/health")
print("Step 5: got health", r.status_code, flush=True)
