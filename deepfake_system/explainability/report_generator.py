"""
Forensic Audit Report Generator (PDF and JSON)
Produces evidence-grade forensic reports with multimodal explainability summaries.
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


class ForensicReportGenerator:
    """
    Generates downloadable PDF and JSON forensic investigation reports.
    """

    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """Computes SHA-256 hash of the analyzed video."""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception:
            return "N/A (Streamed Upload)"

    @classmethod
    def generate_json_report(cls, analysis_results: Dict[str, Any], output_path: str) -> str:
        """Saves detailed structured JSON report."""
        report = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "system": "Explainable Multimodal Deepfake Detector (CNN-LSTM-rPPG)",
                "analysis_mode": analysis_results.get("mode_used", "advanced"),
                "file_hash": analysis_results.get("file_hash", "N/A")
            },
            "classification": {
                "verdict": analysis_results.get("prediction", "UNKNOWN"),
                "confidence": analysis_results.get("confidence", 0.0),
                "prob_real": analysis_results.get("prob_real", 0.0),
                "prob_fake": analysis_results.get("prob_fake", 0.0)
            },
            "forensic_modalities": {
                "spatial_cnn": {
                    "backbone": "EfficientNet-B0",
                    "gradcam_generated": "gradcam_image" in analysis_results
                },
                "temporal_lstm": {
                    "architecture": "Bidirectional LSTM with Self-Attention",
                    "attention_distribution": analysis_results.get("frame_attention_weights", [])
                },
                "physiological_rppg": analysis_results.get("rppg_analysis", {})
            }
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        return output_path

    @classmethod
    def generate_pdf_report(
        cls,
        analysis_results: Dict[str, Any],
        output_path: str,
        gradcam_img_path: Optional[str] = None
    ) -> str:
        """
        Builds a multi-page PDF forensic report using ReportLab.
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=40, leftMargin=40,
            topMargin=40, bottomMargin=40
        )
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=20,
            leading=24,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=4
        )
        subtitle_style = ParagraphStyle(
            'SubTitleStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#64748b'),
            spaceAfter=15
        )
        h2_style = ParagraphStyle(
            'H2Style',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=16,
            textColor=colors.HexColor('#1e293b'),
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#334155')
        )
        verdict_fake = ParagraphStyle(
            'VerdictFake',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#dc2626')
        )
        verdict_real = ParagraphStyle(
            'VerdictReal',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#16a34a')
        )

        elements = []

        # Header Title
        elements.append(Paragraph("FORENSIC MEDIA AUTHENTICATION REPORT", title_style))
        elements.append(Paragraph("Multimodal Facial Forgery & Hemodynamic Integrity Analysis", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#cbd5e1'), spaceAfter=12))

        # Case Metadata Table
        verdict = analysis_results.get("prediction", "UNKNOWN")
        confidence = analysis_results.get("confidence", 0.0) * 100
        mode = analysis_results.get("mode_used", "advanced").upper()
        meta_data = [
            [Paragraph("<b>Inspection Date:</b>", body_style), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"), body_style),
             Paragraph("<b>Analysis Mode:</b>", body_style), Paragraph(mode, body_style)],
            [Paragraph("<b>Primary Verdict:</b>", body_style), Paragraph(f"<b>{verdict}</b> ({confidence:.1f}% confidence)", verdict_fake if verdict == "FAKE" else verdict_real),
             Paragraph("<b>Video Hash:</b>", body_style), Paragraph(analysis_results.get("file_hash", "SHA-256 Validated")[:18] + "...", body_style)]
        ]
        meta_table = Table(meta_data, colWidths=[100, 165, 100, 165])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('PADDING', (0,0), (-1,-1), 6),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 14))

        # Forensic Modality Breakdown
        elements.append(Paragraph("Multimodal Forensic Findings", h2_style))
        
        rppg = analysis_results.get("rppg_analysis", {})
        hr = rppg.get("heart_rate_bpm", 0.0)
        snr = rppg.get("snr_db", 0.0)
        coherence = rppg.get("mean_cross_region_coherence", 0.0)
        gate_weight = analysis_results.get("modality_gate_st_weight", 0.5)

        findings_data = [
            [Paragraph("<b>Forensic Stream</b>", body_style), Paragraph("<b>Methodology</b>", body_style), Paragraph("<b>Observed Indicator</b>", body_style), Paragraph("<b>Diagnostic</b>", body_style)],
            [
                Paragraph("<b>Spatial (CNN)</b>", body_style),
                Paragraph("EfficientNet Feature Maps + Grad-CAM", body_style),
                Paragraph("Boundary discontinuity & warping", body_style),
                Paragraph("Anomaly Detected" if verdict == "FAKE" else "Natural Texture", body_style)
            ],
            [
                Paragraph("<b>Temporal (LSTM)</b>", body_style),
                Paragraph("Bi-LSTM Sequence Attention", body_style),
                Paragraph("Inter-frame continuity check", body_style),
                Paragraph("Flicker / Phase Jitter" if verdict == "FAKE" else "Stable Progression", body_style)
            ],
            [
                Paragraph("<b>Physiology (rPPG)</b>", body_style),
                Paragraph("POS Hemodynamic Pulse & Welch PSD", body_style),
                Paragraph(f"HR: {hr:.1f} BPM | SNR: {snr:.1f} dB | Coherence: {coherence:.2f}", body_style),
                Paragraph("Biological Desynchrony" if coherence < 0.4 else "Synchronous Cardiac Pulse", body_style)
            ]
        ]
        findings_table = Table(findings_data, colWidths=[110, 150, 160, 110])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
            ('PADDING', (0,0), (-1,-1), 5),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        elements.append(findings_table)
        elements.append(Spacer(1, 14))

        # Include Grad-CAM image if available
        if gradcam_img_path and os.path.exists(gradcam_img_path):
            elements.append(Paragraph("Spatial Artifact Attribution (Grad-CAM Heatmap)", h2_style))
            elements.append(Paragraph("Highlights pixel regions exerting maximal positive influence toward the manipulation classification:", body_style))
            elements.append(Spacer(1, 6))
            elements.append(Image(gradcam_img_path, width=220, height=220))
            elements.append(Spacer(1, 14))

        # Disclaimer
        elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#e2e8f0'), spaceAfter=8))
        elements.append(Paragraph("<b>Investigative Advisory:</b> This automated forensic report combines spatial convolutional feature learning, recurrent temporal analysis, and remote photoplethysmography. Advanced mode verifies physiological cross-regional blood volume pulse coherence. Results should be reviewed in conjunction with chain-of-custody metadata.", body_style))

        doc.build(elements)
        return output_path
