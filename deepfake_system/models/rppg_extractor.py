"""
rPPG Signal Extraction and Physiological Forensic Analyzer
Implements POS (Plane-Orthogonal-to-Skin) and CHROM (Chrominance) Algorithms
with Cross-Regional Hemodynamic Coherence Analysis.
"""

import numpy as np
import scipy.signal as signal
from scipy.stats import entropy
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn

from deepfake_system.config import (
    FPS_DEFAULT, RPPG_LOW_FREQ, RPPG_HIGH_FREQ, FILTER_ORDER, RPPG_FEATURE_DIM
)


class RPPGExtractor:
    """
    Deterministic Remote Photoplethysmography (rPPG) Processor.
    Extracts Blood Volume Pulse (BVP) from facial skin color variations and
    computes physiological forensic descriptors (SNR, peak coherence, spectral entropy).
    """

    def __init__(self, fps: float = FPS_DEFAULT, method: str = "POS"):
        self.fps = fps
        self.method = method.upper()
        # Build Butterworth bandpass filter coefficients
        nyquist = 0.5 * self.fps
        low = RPPG_LOW_FREQ / nyquist
        high = min(RPPG_HIGH_FREQ / nyquist, 0.95)
        self.b, self.a = signal.butter(FILTER_ORDER, [low, high], btype="bandpass")

    def _pos_algorithm(self, rgb_series: np.ndarray) -> np.ndarray:
        """
        Plane-Orthogonal-to-Skin (POS) Algorithm (Wang et al., 2017).
        rgb_series: shape (N, 3) where columns are R, G, B temporal traces.
        Returns: 1D array of length N (raw BVP signal).
        """
        N = len(rgb_series)
        if N < 8:
            return np.zeros(N, dtype=np.float32)

        # Temporal mean normalization: c_n = C(t) / mean(C)
        mean_rgb = np.mean(rgb_series, axis=0, keepdims=True) + 1e-6
        norm_rgb = rgb_series / mean_rgb

        # Orthogonal projections
        S1 = norm_rgb[:, 1] - norm_rgb[:, 2]               # G - B
        S2 = norm_rgb[:, 1] + norm_rgb[:, 2] - 2 * norm_rgb[:, 0]  # G + B - 2R

        # Tuning coefficient alpha = std(S1) / std(S2)
        std_s1 = np.std(S1) + 1e-8
        std_s2 = np.std(S2) + 1e-8
        alpha = std_s1 / std_s2

        # BVP pulse signal
        H = S1 + alpha * S2
        return H

    def _chrom_algorithm(self, rgb_series: np.ndarray) -> np.ndarray:
        """
        Chrominance-based (CHROM) Algorithm (de Haan & Jeanne, 2013).
        rgb_series: shape (N, 3).
        """
        N = len(rgb_series)
        if N < 8:
            return np.zeros(N, dtype=np.float32)

        mean_rgb = np.mean(rgb_series, axis=0, keepdims=True) + 1e-6
        norm_rgb = rgb_series / mean_rgb

        X = 3 * norm_rgb[:, 0] - 2 * norm_rgb[:, 1]
        Y = 1.5 * norm_rgb[:, 0] + norm_rgb[:, 1] - 1.5 * norm_rgb[:, 2]

        alpha = (np.std(X) + 1e-8) / (np.std(Y) + 1e-8)
        return X - alpha * Y

    def extract_bvp(self, rgb_series: np.ndarray) -> np.ndarray:
        """
        Extracts and bandpass-filters BVP waveform from an RGB temporal trace.
        """
        if self.method == "CHROM":
            raw_bvp = self._chrom_algorithm(rgb_series)
        else:
            raw_bvp = self._pos_algorithm(rgb_series)

        # Apply zero-phase forward-backward Butterworth filter
        if len(raw_bvp) >= 3 * (FILTER_ORDER + 1):
            try:
                filtered_bvp = signal.filtfilt(self.b, self.a, raw_bvp)
            except Exception:
                filtered_bvp = raw_bvp - np.mean(raw_bvp)
        else:
            filtered_bvp = raw_bvp - np.mean(raw_bvp)

        # Normalize to zero-mean, unit-variance
        std_val = np.std(filtered_bvp) + 1e-6
        return (filtered_bvp - np.mean(filtered_bvp)) / std_val

    def analyze_spectrum(self, bvp_signal: np.ndarray) -> Dict[str, float]:
        """
        Computes Fourier Power Spectral Density (PSD) and biological metrics:
        - Heart Rate (BPM)
        - Signal-to-Noise Ratio (SNR) in cardiac band
        - Peak Power Ratio
        - Spectral Entropy
        """
        N = len(bvp_signal)
        if N < 16:
            return {
                "heart_rate_bpm": 72.0,
                "snr_db": -5.0,
                "peak_power_ratio": 0.1,
                "spectral_entropy": 0.95,
                "dominant_freq_hz": 1.2
            }

        # Welch Power Spectral Density or FFT
        freqs, psd = signal.welch(bvp_signal, fs=self.fps, nperseg=min(N, 128), nfft=256)

        # Restrict to human cardiac frequency range: [0.7 Hz, 3.5 Hz]
        valid_idx = (freqs >= RPPG_LOW_FREQ) & (freqs <= RPPG_HIGH_FREQ)
        if not np.any(valid_idx):
            return {
                "heart_rate_bpm": 72.0, "snr_db": -5.0,
                "peak_power_ratio": 0.1, "spectral_entropy": 0.95,
                "dominant_freq_hz": 1.2
            }

        cardiac_freqs = freqs[valid_idx]
        cardiac_psd = psd[valid_idx]

        # Find dominant peak
        max_idx = np.argmax(cardiac_psd)
        dominant_freq = cardiac_freqs[max_idx]
        heart_rate_bpm = dominant_freq * 60.0

        # Peak power vs total band power
        peak_power = cardiac_psd[max_idx]
        total_power = np.sum(cardiac_psd) + 1e-8
        peak_power_ratio = float(peak_power / total_power)

        # SNR calculation: Fundamental peak power +/- 0.15 Hz divided by noise
        peak_window = (cardiac_freqs >= dominant_freq - 0.15) & (cardiac_freqs <= dominant_freq + 0.15)
        signal_power = np.sum(cardiac_psd[peak_window])
        noise_power = total_power - signal_power + 1e-8
        snr_db = float(10 * np.log10(max(signal_power / noise_power, 1e-4)))

        # Normalized spectral entropy (measure of signal randomness)
        norm_psd = cardiac_psd / (np.sum(cardiac_psd) + 1e-8)
        spec_entropy = float(entropy(norm_psd) / np.log(len(cardiac_psd) + 1e-8))

        return {
            "heart_rate_bpm": float(heart_rate_bpm),
            "snr_db": float(snr_db),
            "peak_power_ratio": float(peak_power_ratio),
            "spectral_entropy": float(spec_entropy),
            "dominant_freq_hz": float(dominant_freq),
            "freqs": freqs[valid_idx].tolist(),
            "psd": norm_psd.tolist()
        }

    def compute_multi_region_coherence(
        self, forehead_rgb: np.ndarray, left_cheek_rgb: np.ndarray, right_cheek_rgb: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyzes biological synchronicity across different facial regions:
        Forehead vs Left Cheek vs Right Cheek.
        Returns cross-correlation coefficients.
        """
        bvp_fh = self.extract_bvp(forehead_rgb)
        bvp_lc = self.extract_bvp(left_cheek_rgb)
        bvp_rc = self.extract_bvp(right_cheek_rgb)

        def _corr(a, b):
            if len(a) < 4 or np.std(a) < 1e-5 or np.std(b) < 1e-5:
                return 0.0
            r = np.corrcoef(a, b)[0, 1]
            return float(np.nan_to_num(r, nan=0.0))

        r_fh_lc = _corr(bvp_fh, bvp_lc)
        r_fh_rc = _corr(bvp_fh, bvp_rc)
        r_lc_rc = _corr(bvp_lc, bvp_rc)
        mean_cross_corr = float((r_fh_lc + r_fh_rc + r_lc_rc) / 3.0)

        # Spectral analysis on composite face signal
        composite_bvp = (bvp_fh + bvp_lc + bvp_rc) / 3.0
        spectral_metrics = self.analyze_spectrum(composite_bvp)

        return {
            "cross_corr_fh_lc": r_fh_lc,
            "cross_corr_fh_rc": r_fh_rc,
            "cross_corr_lc_rc": r_lc_rc,
            "mean_cross_region_coherence": mean_cross_corr,
            **spectral_metrics,
            "bvp_waveform": composite_bvp.tolist()
        }

    def extract_feature_vector(
        self, forehead_rgb: np.ndarray, left_cheek_rgb: np.ndarray, right_cheek_rgb: np.ndarray
    ) -> np.ndarray:
        """
        Transforms multi-region rPPG signals into a fixed 32-D physiological feature vector:
        - 4 Coherence metrics (cross-region correlations)
        - 4 Spectral metrics (SNR, Peak Ratio, Entropy, Dominant Freq)
        - 8 Statistical moments per region (mean, std, skewness, kurtosis)
        - 16 Interpolated PSD power bins in cardiac range [0.7, 3.5 Hz]
        """
        res = self.compute_multi_region_coherence(forehead_rgb, left_cheek_rgb, right_cheek_rgb)

        # Multi-region coherence & spectrum
        feat = [
            res["cross_corr_fh_lc"],
            res["cross_corr_fh_rc"],
            res["cross_corr_lc_rc"],
            res["mean_cross_region_coherence"],
            res["snr_db"] / 20.0,                   # scaled roughly [-1, 1]
            res["peak_power_ratio"],
            res["spectral_entropy"],
            (res["heart_rate_bpm"] - 75.0) / 40.0   # standardized HR
        ]

        # Moments of BVP waveform
        bvp = np.array(res["bvp_waveform"])
        if len(bvp) > 4:
            feat.extend([
                float(np.mean(bvp)),
                float(np.std(bvp)),
                float(np.percentile(bvp, 75) - np.percentile(bvp, 25)),
                float(np.max(bvp) - np.min(bvp)),
                float(np.mean(np.diff(bvp) ** 2)),  # High frequency derivative energy
                float(np.mean(np.abs(np.diff(bvp)))),
                float(np.sum(np.diff(np.sign(np.diff(bvp))) != 0) / max(len(bvp), 1)), # Zero-crossing rate
                float(np.var(np.diff(bvp)))
            ])
        else:
            feat.extend([0.0] * 8)

        # 16-bin normalized power spectrum in [0.7, 3.5 Hz]
        if "psd" in res and len(res["psd"]) > 0:
            psd_vals = np.array(res["psd"])
            # Resample psd_vals to exactly 16 bins
            x_old = np.linspace(0, 1, len(psd_vals))
            x_new = np.linspace(0, 1, 16)
            psd_16 = np.interp(x_new, x_old, psd_vals)
            feat.extend(psd_16.tolist())
        else:
            feat.extend([0.0] * 16)

        feat_arr = np.array(feat[:RPPG_FEATURE_DIM], dtype=np.float32)
        # Pad with zeros if necessary to strictly match RPPG_FEATURE_DIM
        if len(feat_arr) < RPPG_FEATURE_DIM:
            feat_arr = np.pad(feat_arr, (0, RPPG_FEATURE_DIM - len(feat_arr)))
        return feat_arr
