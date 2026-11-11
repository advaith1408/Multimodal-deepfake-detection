# Changes

Log of changes made to this project in the current working session, plus the latency measurements and the logic review that came out of it.

## 1. Changes made

### Rate limiting (new)
- **`deepfake_system/api/rate_limit.py`** (new): dependency-free, in-memory sliding-window limiter, keyed by client IP.
- **`deepfake_system/api/app.py`**: limits wired to the endpoints; `/api/predict` body moved into `_analyze_video()` and wrapped in a concurrency slot.

| Endpoint | Limit (per client / minute) | Env var |
|---|---|---|
| `POST /api/predict` | 5 | `RATE_LIMIT_PREDICT` |
| `POST /api/report/pdf` | 20 | `RATE_LIMIT_REPORT` |
| `GET /api/benchmark` | 60 | `RATE_LIMIT_DEFAULT` |

- Exceeding a limit returns `429` with `Retry-After`.
- Concurrent analyses are capped (`MAX_CONCURRENT_PREDICTIONS`, **default 1**); extra requests get `503` + `Retry-After`. Default is 1 because the detector holds shared mutable state (see flaw #7).
- Other switches: `RATE_LIMIT_ENABLED=0` disables everything; `TRUST_PROXY_HEADERS=1` keys clients by `X-Forwarded-For` (only behind a trusted proxy).
- Limitation: state is per-process; multiple workers need a shared store (e.g. Redis).

### UI redesign
- **`static/index.html`** rewritten; **`static/style.css`** (new); **`static/app.js`** updated to match.
- Plain CSS replaces the Tailwind CDN. Only the Inter font and Lucide icons load from the network.
- Palette: warm neutrals + a single sage accent (`#7fb8a4`). No other hues. Authentic = accent; fake = inverted off-white state (icon, border, title).
- Strict grid layout: equal-height cards, aligned headers/footers, three-column verdict banner, responsive below 960px.
- Quiet 200ms transitions, hover states, staged fade-in, animated confidence bar; all disabled under `prefers-reduced-motion`.
- Verdict state is driven by `data-state="real|fake"` on the card instead of swapped class strings.
- Fixed: "Load Diffusion AI" button had no click handler. Drop zone is keyboard-accessible.

### Environment
- Installed: `opencv-python<5` (5.x removed `CascadeClassifier`), `numpy`, `scipy`, `scikit-learn`, `reportlab`, `python-multipart`, `torchvision`, `timm`.
- Note: pip warned `click` 8.5.1 conflicts with the unrelated `gtts` package.
- Repo was cloned from `https://github.com/advaith1408/Multimodal-deepfake-detection.git`.

## 2. Latency (CPU, 14 threads, no GPU)

End-to-end `POST /api/predict`, measured through the running server:

| Sample | Resolution / frames | Standard | Advanced |
|---|---|---|---|
| sample_real | 1920x1080, 211 | 24.6 s | 26.2 s |
| sample_diffusion | 1280x720, 240 | 17.7 s | 25.9 s |
| sample_fake | 320x320, 75 | 2.7 s | 5.5 s |

Stage breakdown (profiled separately):

| Stage | real (1080p) | diffusion (720p) | fake (320p) |
|---|---|---|---|
| Decode up to 300 frames | 2.3 s | 1.8 s | 0.2 s |
| **Face tracking (Haar on every frame)** | **39.3 s** | **27.8 s** | **1.6 s** |
| CNN + LSTM (16 frames) | 1.3 s | 0.8 s | 1.3 s |
| Spatial forensics | 0.09 s | 0.06 s | 0.07 s |
| rPPG | 0.03 s | 0.01 s | 0.00 s |

Other: model load 0.5 s; `/api/health` ~5 ms; `/api/benchmark` ~4 ms (hard-coded, see flaw #3).

**Bottleneck:** `FaceAndROIDetector.track_video_sequence` runs two Haar cascades on every full-resolution frame (up to 300). It accounts for roughly 90%+ of latency on HD input; the neural network is only ~1 s. Cost scales with pixel count.

**Cheap fixes (not applied):**
1. Run detection on frames downscaled to ~480 px wide and scale boxes back up.
2. Detect on every 3rd–5th frame (or only the 16 sampled + needed rPPG frames) and interpolate the rest; the interpolation code already exists.
3. Stop decoding at the frames actually used; cap/resize on upload.
4. Replace Haar with a lighter DNN detector if accuracy on non-frontal faces matters.

Expected effect: HD videos from ~25 s down to a few seconds.

## 3. Logic review: flaws found

Ordered by severity.

1. **The neural network does not decide the verdict; hard-coded rules do.** In `predict_video` the model's probabilities are overwritten by fixed thresholds on Laplacian variance, FFT energy and edge gradient. Measured raw network P(fake) vs final result:

   | Sample | Network P(fake) std / adv | Final verdict |
   |---|---|---|
   | sample_fake | 0.004 / 0.030 | FAKE 0.963 |
   | sample_real | 0.004 / 0.050 | REAL |
   | sample_diffusion | 0.003 / 0.042 | FAKE 0.930 |

   The network outputs "real" for everything, including the fake and diffusion clips, so the classifier has effectively collapsed. The correct-looking verdicts come only from thresholds that appear tuned to these sample files (`lap < 65`, `grad > 72`, `lap >= 85`), which will not generalise: a sharp FaceSwap with soft seams is "REAL"; a soft or heavily compressed genuine video is "FAKE"; a blurred phone video is "FAKE".
2. **Confidence values are fabricated.** In the "natural sensor" branch standard mode returns a constant 0.92 real probability, and the other branches clamp to preset ranges (`max(0.93, ...)`, `max(0.95, ...)`). The UI presents these as model confidence.
3. **Benchmark results are hard-coded.** `/api/benchmark` returns fixed accuracy/F1/AUC numbers and ROC curves built from `np.random.beta` draws; the modal table in the UI is also static HTML. They are labelled "FaceForensics++ C23" but were not computed from any evaluation run. This is misleading and should be replaced with real evaluation output or clearly marked illustrative.
4. **The rPPG branch is effectively a no-op.** The condition `(coherence >= 0.15 and 40 <= hr <= 180) or (coherence >= 0.0)` makes the first clause redundant and ignores heart rate; both outcomes still produce REAL with ≥0.90. rPPG cannot flip a verdict toward FAKE. Also, clips are as short as 2.5 s (75 frames), giving ~0.4 Hz spectral resolution, so heart rate is unreliable, and only the first 128 frames are used. `RPPG_WINDOW_LENGTH` in config is unused.
5. **No "no face found" outcome.** If the Haar cascades never fire, the tracker falls back to a fixed centre crop and the pipeline still returns REAL/FAKE with high confidence on a non-face. Multiple faces are resolved by "largest box in the upper 65% of the frame" with no warning.
6. **Grad-CAM explains the wrong thing.** The target class is the final (rule-overridden) label, not the CNN's own prediction, so the heatmap is conditioned on a verdict the network did not produce.
7. **Shared mutable state is not thread-safe.** One global detector holds `rppg_extractor.fps`/filter coefficients, the face tracker's state, and Grad-CAM forward/backward hooks. `/api/predict` runs in a thread pool, so concurrent requests can corrupt each other's results. Mitigated by setting the concurrency cap default to 1 (this change); a lock or per-request state would be the real fix.
8. **Training data and setup are too small and not portable.** `training_data/` holds 32 short synthetic videos; `train_model.py` and the scripts reference hard-coded paths under `C:\Users\advai\Downloads\...`. A model trained on this cannot be expected to generalise, which is consistent with flaw #1. There is no `requirements.txt`.
9. **Upload handling.**
   - The temp file is named from the client-supplied `video.filename` (collisions between same-named concurrent uploads; unsanitised path component).
   - The "50 MB" limit in the UI is not enforced server-side; there is no content-type check.
   - `/api/report/pdf` generates a "forensic report" from arbitrary client-posted JSON (reports can be forged), writes a fixed `gradcam_temp.png` (race), and never deletes the generated PDFs (temp-dir leak).
10. **Smaller issues.** CORS is `*` with credentials; sampled crops use `num_frames // seq_len` steps so the last part of long videos is never sampled; the 5 req/min limit will trip the existing eval scripts if they loop over `/api/predict`.

## 4. Suggested next steps
1. Fix latency (downscale + frame skipping in the tracker): biggest win, low risk.
2. Decide what the system should claim: either make the CNN/LSTM/rPPG model drive the decision (retrain on a real dataset, e.g. FaceForensics++, and report real metrics), or present the output honestly as a heuristic screening tool.
3. Replace the hard-coded benchmark with a real evaluation, or label it illustrative.
4. Add a "no face detected" result and per-request state/locking.
5. Sanitise uploads, enforce size limits, delete generated PDFs, restrict CORS, add `requirements.txt`.
