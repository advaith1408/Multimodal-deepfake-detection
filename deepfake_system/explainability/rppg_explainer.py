"""
Physiological & Temporal Explainability Visualizer
Generates interactive plots and base64 images for rPPG waveforms, PSD spectra, and attention.
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
from typing import Dict, List, Any


class PhysiologicalExplainer:
    """
    Renders explainability figures for biological signal validity and temporal attention.
    """

    @staticmethod
    def plot_rppg_analysis(rppg_data: Dict[str, Any]) -> str:
        """
        Plots BVP pulse waveform and Power Spectral Density (PSD).
        Returns: base64-encoded PNG image string.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.5), dpi=100)
        fig.patch.set_facecolor('#0f172a')  # Dark theme to match UI

        # Plot 1: BVP Waveform
        waveform = rppg_data.get("bvp_waveform", [])
        time_axis = np.linspace(0, len(waveform) / 30.0, len(waveform)) if len(waveform) > 0 else []

        ax1.set_facecolor('#1e293b')
        if len(waveform) > 0:
            ax1.plot(time_axis, waveform, color='#38bdf8', linewidth=1.8, label="Blood Volume Pulse (POS)")
        ax1.set_title("Facial Hemodynamic Pulse Waveform (rPPG)", color='#f8fafc', fontsize=11, fontweight='bold')
        ax1.set_xlabel("Time (seconds)", color='#94a3b8', fontsize=9)
        ax1.set_ylabel("Normalized Amplitude", color='#94a3b8', fontsize=9)
        ax1.tick_params(colors='#94a3b8', labelsize=8)
        ax1.grid(True, linestyle='--', alpha=0.2, color='#64748b')
        ax1.legend(loc="upper right", facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8)

        # Plot 2: Power Spectral Density (PSD)
        freqs = rppg_data.get("freqs", [])
        psd = rppg_data.get("psd", [])
        hr_bpm = rppg_data.get("heart_rate_bpm", 0.0)
        snr = rppg_data.get("snr_db", 0.0)

        ax2.set_facecolor('#1e293b')
        if len(freqs) > 0 and len(psd) > 0:
            ax2.plot(freqs, psd, color='#ec4899', linewidth=1.8, label=f"PSD (Estimated HR: {hr_bpm:.1f} BPM)")
            ax2.axvline(x=hr_bpm / 60.0, color='#f43f5e', linestyle=':', linewidth=2, label=f"Cardiac Fundamental ({hr_bpm/60.0:.2f} Hz)")
        ax2.set_title(f"Frequency Spectrum Analysis (SNR: {snr:.1f} dB)", color='#f8fafc', fontsize=11, fontweight='bold')
        ax2.set_xlabel("Frequency (Hz)", color='#94a3b8', fontsize=9)
        ax2.set_ylabel("Power Density", color='#94a3b8', fontsize=9)
        ax2.tick_params(colors='#94a3b8', labelsize=8)
        ax2.grid(True, linestyle='--', alpha=0.2, color='#64748b')
        ax2.legend(loc="upper right", facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def plot_temporal_attention(attn_weights: List[float]) -> str:
        """
        Plots frame-level attention distribution highlighting temporal anomalies.
        Returns: base64-encoded PNG image string.
        """
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')

        frames = np.arange(1, len(attn_weights) + 1)
        bars = ax.bar(frames, attn_weights, color='#6366f1', edgecolor='#818cf8', alpha=0.85)

        # Highlight highest anomaly frame
        if len(attn_weights) > 0:
            max_idx = np.argmax(attn_weights)
            bars[max_idx].set_color('#ef4444')
            bars[max_idx].set_edgecolor('#fca5a5')

        ax.set_title("LSTM Temporal Attention Weighting (Red = Peak Anomaly Frame)", color='#f8fafc', fontsize=10, fontweight='bold')
        ax.set_xlabel("Sampled Sequence Frame Index", color='#94a3b8', fontsize=9)
        ax.set_ylabel("Attention Score", color='#94a3b8', fontsize=9)
        ax.tick_params(colors='#94a3b8', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.2, color='#64748b')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
