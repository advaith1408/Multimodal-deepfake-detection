"""
Comparative Evaluation Suite: CNN vs. CNN+LSTM vs. CNN+LSTM+rPPG
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import io
import base64


class ModelEvaluator:
    """
    Rigorously evaluates and compares the three architectural configurations:
    1. CNN Only (Spatial frame classification)
    2. CNN + LSTM (Standard Mode: Spatial-Temporal)
    3. CNN + LSTM + rPPG (Advanced Mode: Multimodal Fusion)
    """

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Computes standard forensic classification metrics.
        """
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5

        cm = confusion_matrix(y_true, y_pred).tolist()

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm
        }

    @classmethod
    def plot_comparative_roc(
        cls,
        eval_dict: Dict[str, Dict[str, np.ndarray]]
    ) -> str:
        """
        Plots ROC Curves comparing the 3 configurations.
        eval_dict format:
        {
           'CNN Only': {'y_true': [...], 'y_prob': [...]},
           'CNN + LSTM': {'y_true': [...], 'y_prob': [...]},
           'CNN + LSTM + rPPG': {'y_true': [...], 'y_prob': [...]}
        }
        Returns: base64-encoded PNG image string.
        """
        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')

        colors_map = {
            'CNN Only': '#94a3b8',
            'CNN + LSTM (Standard)': '#38bdf8',
            'CNN + LSTM + rPPG (Advanced)': '#ec4899'
        }

        for name, data in eval_dict.items():
            y_t = np.array(data["y_true"])
            y_p = np.array(data["y_prob"])
            try:
                fpr, tpr, _ = roc_curve(y_t, y_p)
                auc_val = roc_auc_score(y_t, y_p)
                color = colors_map.get(name, '#6366f1')
                ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{name} (AUC = {auc_val:.3f})")
            except Exception:
                continue

        # Diagonal chance line
        ax.plot([0, 1], [0, 1], linestyle='--', color='#475569', label='Random Guess (AUC = 0.500)')
        ax.set_title("Comparative ROC Curves (FaceForensics++ Evaluation)", color='#f8fafc', fontsize=12, fontweight='bold')
        ax.set_xlabel("False Positive Rate", color='#94a3b8', fontsize=10)
        ax.set_ylabel("True Positive Rate", color='#94a3b8', fontsize=10)
        ax.tick_params(colors='#94a3b8', labelsize=9)
        ax.grid(True, linestyle='--', alpha=0.25, color='#64748b')
        ax.legend(loc="lower right", facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

    @classmethod
    def plot_confusion_matrix(cls, cm: List[List[int]], title: str = "Confusion Matrix") -> str:
        """
        Renders confusion matrix graphic.
        """
        fig, ax = plt.subplots(figsize=(4.5, 4), dpi=100)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')

        cm_arr = np.array(cm)
        im = ax.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(title, color='#f8fafc', fontsize=11, fontweight='bold')

        classes = ['REAL (0)', 'FAKE (1)']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, color='#94a3b8', fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, color='#94a3b8', fontsize=9)

        # Labels inside cells
        thresh = cm_arr.max() / 2.0
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(j, i, format(cm_arr[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm_arr[i, j] > thresh else "#cbd5e1",
                        fontsize=12, fontweight='bold')

        ax.set_ylabel('True Forensic Label', color='#94a3b8', fontsize=10)
        ax.set_xlabel('Predicted Label', color='#94a3b8', fontsize=10)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
