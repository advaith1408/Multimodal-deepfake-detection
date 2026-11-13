"""
Physiological & Temporal Explainability Visualizer
Generates interactive plots and base64 images for rPPG waveforms, PSD spectra, and attention.
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
from matplotlib.figure import Figure
from typing import Dict, List, Any


class PhysiologicalExplainer:
    """
    Renders explainability figures for biological signal validity and temporal attention
    using thread-safe Matplotlib Figure instances in a monochrome Black & White theme.
    """

    @staticmethod
    def plot_rppg_analysis(rppg_data: Dict[str, Any]) -> str:
        """
        Plots BVP pulse waveform and Power Spectral Density (PSD).
        Returns: base64-encoded PNG image string.
        """
        fig = Figure(figsize=(8, 4.5), dpi=100)
        fig.patch.set_facecolor('#000000')
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        # Plot 1: BVP Waveform
        waveform = rppg_data.get("bvp_waveform", [])
        time_axis = np.linspace(0, len(waveform) / 30.0, len(waveform)) if len(waveform) > 0 else []

        ax1.set_facecolor('#0a0a0a')
        if len(waveform) > 0:
            ax1.plot(time_axis, waveform, color='#ffffff', linewidth=1.8, label="Blood Volume Pulse (POS)")
        ax1.set_title("Facial Hemodynamic Pulse Waveform (rPPG)", color='#ffffff', fontsize=11, fontweight='bold')
        ax1.set_xlabel("Time (seconds)", color='#a3a3a3', fontsize=9)
        ax1.set_ylabel("Normalized Amplitude", color='#a3a3a3', fontsize=9)
        ax1.tick_params(colors='#a3a3a3', labelsize=8)
        for spine in ax1.spines.values():
            spine.set_edgecolor('#262626')
        ax1.grid(True, linestyle='--', alpha=0.25, color='#525252')
        ax1.legend(loc="upper right", facecolor='#0a0a0a', edgecolor='#404040', labelcolor='#e5e5e5', fontsize=8)

        # Plot 2: Power Spectral Density (PSD)
        freqs = rppg_data.get("freqs", [])
        psd = rppg_data.get("psd", [])
        hr_bpm = rppg_data.get("heart_rate_bpm", 0.0)
        snr = rppg_data.get("snr_db", 0.0)

        ax2.set_facecolor('#0a0a0a')
        if len(freqs) > 0 and len(psd) > 0:
            ax2.plot(freqs, psd, color='#d4d4d4', linewidth=1.8, label=f"PSD (Estimated HR: {hr_bpm:.1f} BPM)")
            ax2.axvline(x=hr_bpm / 60.0, color='#ffffff', linestyle=':', linewidth=2, label=f"Cardiac Fundamental ({hr_bpm/60.0:.2f} Hz)")
        ax2.set_title(f"Frequency Spectrum Analysis (SNR: {snr:.1f} dB)", color='#ffffff', fontsize=11, fontweight='bold')
        ax2.set_xlabel("Frequency (Hz)", color='#a3a3a3', fontsize=9)
        ax2.set_ylabel("Power Density", color='#a3a3a3', fontsize=9)
        ax2.tick_params(colors='#a3a3a3', labelsize=8)
        for spine in ax2.spines.values():
            spine.set_edgecolor('#262626')
        ax2.grid(True, linestyle='--', alpha=0.25, color='#525252')
        ax2.legend(loc="upper right", facecolor='#0a0a0a', edgecolor='#404040', labelcolor='#e5e5e5', fontsize=8)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def plot_temporal_attention(attn_weights: List[float]) -> str:
        """
        Plots frame-level attention distribution highlighting temporal anomalies.
        Returns: base64-encoded PNG image string.
        """
        fig = Figure(figsize=(8, 2.5), dpi=100)
        fig.patch.set_facecolor('#000000')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor('#0a0a0a')

        frames = np.arange(1, len(attn_weights) + 1)
        bars = ax.bar(frames, attn_weights, color='#525252', edgecolor='#a3a3a3', alpha=0.9)

        # Highlight highest anomaly frame in bright white
        if len(attn_weights) > 0:
            max_idx = int(np.argmax(attn_weights))
            bars[max_idx].set_color('#ffffff')
            bars[max_idx].set_edgecolor('#ffffff')

        ax.set_title("LSTM Temporal Attention Weighting (White = Peak Anomaly Frame)", color='#ffffff', fontsize=10, fontweight='bold')
        ax.set_xlabel("Sampled Sequence Frame Index", color='#a3a3a3', fontsize=9)
        ax.set_ylabel("Attention Score", color='#a3a3a3', fontsize=9)
        ax.tick_params(colors='#a3a3a3', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#262626')
        ax.grid(True, linestyle='--', alpha=0.25, color='#525252')

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
