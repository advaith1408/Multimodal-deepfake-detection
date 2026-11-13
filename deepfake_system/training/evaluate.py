"""
Comparative Evaluation Suite: CNN vs. CNN+LSTM vs. CNN+LSTM+rPPG
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from matplotlib.figure import Figure
from typing import Dict, List, Any
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

        cm_list = confusion_matrix(y_true, y_pred).tolist()

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm_list
        }

    @classmethod
    def plot_comparative_roc(
        cls,
        eval_dict: Dict[str, Dict[str, np.ndarray]]
    ) -> str:
        """
        Plots ROC Curves comparing the 3 configurations using thread-safe Figure API
        in a monochrome Black & White theme.
        Returns: base64-encoded PNG image string.
        """
        fig = Figure(figsize=(7, 5), dpi=110)
        fig.patch.set_facecolor('#000000')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0a0a0a')

        styles_map = {
            'CNN Only': {'color': '#737373', 'linestyle': ':', 'linewidth': 1.8},
            'CNN + LSTM (Standard)': {'color': '#bcbcbc', 'linestyle': '--', 'linewidth': 2.0},
            'CNN + LSTM + rPPG (Advanced)': {'color': '#ffffff', 'linestyle': '-', 'linewidth': 2.6}
        }

        for name, data in eval_dict.items():
            y_t = np.array(data["y_true"])
            y_p = np.array(data["y_prob"])
            try:
                fpr, tpr, _ = roc_curve(y_t, y_p)
                auc_val = roc_auc_score(y_t, y_p)
                st = styles_map.get(name, {'color': '#ffffff', 'linestyle': '-', 'linewidth': 2.0})
                ax.plot(
                    fpr, tpr,
                    color=st['color'],
                    linestyle=st['linestyle'],
                    linewidth=st['linewidth'],
                    label=f"{name} (AUC = {auc_val:.3f})"
                )
            except Exception:
                continue

        # Diagonal chance line
        ax.plot([0, 1], [0, 1], linestyle='-.', color='#404040', linewidth=1.2, label='Random Guess (AUC = 0.500)')
        ax.set_title("Comparative ROC Curves (FaceForensics++ Evaluation)", color='#ffffff', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("False Positive Rate", color='#a3a3a3', fontsize=10)
        ax.set_ylabel("True Positive Rate", color='#a3a3a3', fontsize=10)
        ax.tick_params(colors='#a3a3a3', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#262626')
        ax.grid(True, linestyle='--', alpha=0.25, color='#525252')
        ax.legend(loc="lower right", facecolor='#0a0a0a', edgecolor='#404040', labelcolor='#e5e5e5', fontsize=8.5)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

    @classmethod
    def plot_confusion_matrix(cls, cm_data: List[List[int]], title: str = "Confusion Matrix") -> str:
        """
        Renders confusion matrix graphic using thread-safe Figure API in Black & White.
        """
        fig = Figure(figsize=(4.8, 4.2), dpi=110)
        fig.patch.set_facecolor('#000000')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0a0a0a')

        cm_arr = np.array(cm_data)
        ax.imshow(cm_arr, interpolation='nearest', cmap=cm.Greys_r)
        ax.set_title(title, color='#ffffff', fontsize=11, fontweight='bold', pad=10)

        classes = ['REAL (0)', 'FAKE (1)']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, color='#a3a3a3', fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, color='#a3a3a3', fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#404040')

        # Labels inside cells
        thresh = cm_arr.max() / 2.0
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(
                    j, i, format(cm_arr[i, j], 'd'),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="#000000" if cm_arr[i, j] > thresh else "#ffffff",
                    fontsize=13, fontweight='bold'
                )

        ax.set_ylabel('True Forensic Label', color='#a3a3a3', fontsize=10)
        ax.set_xlabel('Predicted Label', color='#a3a3a3', fontsize=10)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
