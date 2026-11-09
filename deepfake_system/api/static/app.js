/**
 * Client-Side Application Logic for Explainable Multimodal Deepfake Detector
 */

document.addEventListener("DOMContentLoaded", () => {
  // Initialize Lucide icons
  lucide.createIcons();

  // State
  let currentMode = "advanced";
  let selectedFile = null;
  let lastAnalysisResult = null;

  // DOM Elements
  const modeStandardBtn = document.getElementById("mode-standard");
  const modeAdvancedBtn = document.getElementById("mode-advanced");
  const modeDesc = document.getElementById("mode-description");

  const dropzone = document.getElementById("dropzone");
  const videoInput = document.getElementById("video-input");
  const selectedFileLabel = document.getElementById("selected-file-label");
  const uploadStatus = document.getElementById("upload-status");
  const btnAnalyze = document.getElementById("btn-analyze");

  const btnLoadReal = document.getElementById("btn-load-real");
  const btnLoadFake = document.getElementById("btn-load-fake");
  const btnLoadDiffusion = document.getElementById("btn-load-diffusion");

  const loadingIndicator = document.getElementById("loading-indicator");
  const resultsContainer = document.getElementById("results-container");

  const verdictCard = document.getElementById("verdict-card");
  const verdictIcon = document.getElementById("verdict-icon");
  const verdictIconContainer = document.getElementById("verdict-icon-container");
  const verdictTitle = document.getElementById("verdict-title");
  const verdictModeBadge = document.getElementById("verdict-mode-badge");
  const verdictHash = document.getElementById("verdict-hash");
  const confidencePercentage = document.getElementById("confidence-percentage");
  const confidenceBar = document.getElementById("confidence-bar");
  const probReal = document.getElementById("prob-real");
  const probFake = document.getElementById("prob-fake");
  const modalityGate = document.getElementById("modality-gate");

  const gradcamDisplay = document.getElementById("gradcam-display");
  const gradcamEmpty = document.getElementById("gradcam-empty");
  const attentionChartDisplay = document.getElementById("attention-chart-display");
  const attentionEmpty = document.getElementById("attention-empty");
  const rppgChartDisplay = document.getElementById("rppg-chart-display");
  const rppgEmpty = document.getElementById("rppg-empty");

  const rppgHr = document.getElementById("rppg-hr");
  const rppgSnr = document.getElementById("rppg-snr");
  const rppgCoherence = document.getElementById("rppg-coherence");
  const peakFrameLabel = document.getElementById("peak-frame-label");

  const btnDownloadPdf = document.getElementById("btn-download-pdf");
  const btnDownloadJson = document.getElementById("btn-download-json");

  const btnBenchmarkModal = document.getElementById("btn-benchmark-modal");
  const benchmarkModal = document.getElementById("benchmark-modal");
  const btnCloseModal = document.getElementById("btn-close-modal");
  const modalRocImg = document.getElementById("modal-roc-img");
  const modalCmImg = document.getElementById("modal-cm-img");

  // Mode Selection Handlers
  modeStandardBtn.addEventListener("click", () => {
    currentMode = "standard";
    modeStandardBtn.className = "mode-btn relative p-3 rounded-lg border text-left transition border-brand-500 bg-brand-500/10 ring-1 ring-brand-500";
    modeAdvancedBtn.className = "mode-btn relative p-3 rounded-lg border text-left transition border-slate-700 bg-slate-800/40 hover:bg-slate-800";
    modeDesc.textContent = "* Standard Mode: Evaluates spatial CNN artifacts and temporal sequence consistency (Bi-LSTM). Faster inference.";
  });

  modeAdvancedBtn.addEventListener("click", () => {
    currentMode = "advanced";
    modeAdvancedBtn.className = "mode-btn relative p-3 rounded-lg border text-left transition border-brand-500 bg-brand-500/10 ring-1 ring-brand-500";
    modeStandardBtn.className = "mode-btn relative p-3 rounded-lg border text-left transition border-slate-700 bg-slate-800/40 hover:bg-slate-800";
    modeDesc.textContent = "* Advanced Mode: Evaluates spatial CNN artifacts, temporal Bi-LSTM consistency, AND subcutaneous hemodynamic pulse coherence (rPPG).";
  });

  // Dropzone Handlers
  dropzone.addEventListener("click", () => videoInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("border-brand-500", "bg-slate-800/60");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("border-brand-500", "bg-slate-800/60");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("border-brand-500", "bg-slate-800/60");
    if (e.dataTransfer.files.length > 0) {
      setVideoFile(e.dataTransfer.files[0]);
    }
  });

  videoInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      setVideoFile(e.target.files[0]);
    }
  });

  function setVideoFile(file) {
    selectedFile = file;
    selectedFileLabel.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    uploadStatus.textContent = "Ready for forensic analysis.";
    btnAnalyze.disabled = false;
  }

  // Preloaded Sample Buttons
  async function loadSampleVideo(samplePath, sampleName) {
    uploadStatus.textContent = `Fetching ${sampleName}...`;
    try {
      const resp = await fetch(samplePath);
      const blob = await resp.blob();
      const file = new File([blob], sampleName, { type: "video/mp4" });
      setVideoFile(file);
      uploadStatus.textContent = `Sample loaded: ${sampleName}`;
    } catch (err) {
      uploadStatus.textContent = `Failed to load sample: ${err.message}`;
    }
  }

  btnLoadReal.addEventListener("click", () => loadSampleVideo("/static/samples/sample_real.mp4", "sample_real.mp4"));
  btnLoadFake.addEventListener("click", () => loadSampleVideo("/static/samples/sample_fake.mp4", "sample_fake.mp4"));

  // Primary Forensic Detection Execution
  btnAnalyze.addEventListener("click", async () => {
    if (!selectedFile) return;

    btnAnalyze.disabled = true;
    loadingIndicator.classList.remove("hidden");
    resultsContainer.classList.add("hidden");

    const formData = new FormData();
    formData.append("video", selectedFile);
    formData.append("mode", currentMode);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Forensic analysis failed.");
      }

      const data = await response.json();
      lastAnalysisResult = data;
      renderResults(data);

    } catch (err) {
      alert(`Analysis Error: ${err.message}`);
    } finally {
      loadingIndicator.classList.add("hidden");
      btnAnalyze.disabled = false;
    }
  });

  // Render Forensic Results
  function renderResults(data) {
    resultsContainer.classList.remove("hidden");

    const isFake = data.prediction === "FAKE";
    const conf = (data.confidence * 100).toFixed(1);

    // Verdict Styling
    if (isFake) {
      verdictCard.className = "bg-dark-900 border border-rose-500/40 rounded-xl p-6 shadow-xl relative overflow-hidden ring-1 ring-rose-500/30";
      verdictIconContainer.className = "p-4 rounded-2xl bg-rose-500/10 text-rose-500 border border-rose-500/30";
      verdictTitle.textContent = "SYNTHETIC DEEPFAKE DETECTED";
      verdictTitle.className = "text-3xl font-extrabold tracking-tight mt-1 text-rose-500";
      confidenceBar.className = "h-full bg-rose-500 transition-all duration-700";
      verdictIcon.setAttribute("data-lucide", "alert-triangle");
    } else {
      verdictCard.className = "bg-dark-900 border border-emerald-500/40 rounded-xl p-6 shadow-xl relative overflow-hidden ring-1 ring-emerald-500/30";
      verdictIconContainer.className = "p-4 rounded-2xl bg-emerald-500/10 text-emerald-500 border border-emerald-500/30";
      verdictTitle.textContent = "AUTHENTIC HUMAN FACE";
      verdictTitle.className = "text-3xl font-extrabold tracking-tight mt-1 text-emerald-400";
      confidenceBar.className = "h-full bg-emerald-500 transition-all duration-700";
      verdictIcon.setAttribute("data-lucide", "shield-check");
    }

    verdictModeBadge.textContent = `${data.mode_used.toUpperCase()} MODE`;
    verdictHash.textContent = `SHA-256: ${data.file_hash || 'N/A'}`;
    confidencePercentage.textContent = `${conf}%`;
    confidenceBar.style.width = `${conf}%`;

    probReal.textContent = data.prob_real.toFixed(3);
    probFake.textContent = data.prob_fake.toFixed(3);
    modalityGate.textContent = data.modality_gate_st_weight.toFixed(2);

    // Grad-CAM Display
    if (data.gradcam_image) {
      gradcamDisplay.src = data.gradcam_image;
      gradcamDisplay.classList.remove("hidden");
      gradcamEmpty.classList.add("hidden");
    } else {
      gradcamDisplay.classList.add("hidden");
      gradcamEmpty.classList.remove("hidden");
    }

    // Temporal Attention Display
    if (data.attention_chart) {
      attentionChartDisplay.src = data.attention_chart;
      attentionChartDisplay.classList.remove("hidden");
      attentionEmpty.classList.add("hidden");
    } else {
      attentionChartDisplay.classList.add("hidden");
      attentionEmpty.classList.remove("hidden");
    }
    peakFrameLabel.textContent = `Frame #${data.peak_anomaly_frame_idx + 1}`;

    // rPPG Waveform & Spectrum Display
    if (data.rppg_chart) {
      rppgChartDisplay.src = data.rppg_chart;
      rppgChartDisplay.classList.remove("hidden");
      rppgEmpty.classList.add("hidden");
      
      const r = data.rppg_summary;
      rppgHr.textContent = `${r.heart_rate_bpm} BPM`;
      rppgSnr.textContent = `${r.snr_db} dB`;
      rppgCoherence.textContent = r.cross_region_coherence.toFixed(2);
      rppgCoherence.className = r.cross_region_coherence > 0.4 ? "text-xs font-bold text-emerald-400 font-mono" : "text-xs font-bold text-rose-400 font-mono";
    } else {
      rppgChartDisplay.classList.add("hidden");
      rppgEmpty.classList.remove("hidden");
      rppgHr.textContent = "N/A";
      rppgSnr.textContent = "N/A";
      rppgCoherence.textContent = "N/A";
    }

    lucide.createIcons();
    resultsContainer.scrollIntoView({ behavior: "smooth" });
  }

  // PDF Report Download
  btnDownloadPdf.addEventListener("click", async () => {
    if (!lastAnalysisResult) return;

    btnDownloadPdf.disabled = true;
    btnDownloadPdf.innerHTML = `<i data-lucide="loader-2" class="w-4 h-4 animate-spin text-rose-400"></i><span>Building PDF...</span>`;
    lucide.createIcons();

    try {
      const response = await fetch("/api/report/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(lastAnalysisResult)
      });

      if (!response.ok) throw new Error("PDF generation failed.");

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `Forensic_Deepfake_Report_${lastAnalysisResult.prediction}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert(`PDF Error: ${err.message}`);
    } finally {
      btnDownloadPdf.disabled = false;
      btnDownloadPdf.innerHTML = `<i data-lucide="file-text" class="w-4 h-4 text-rose-400"></i><span>Download PDF Report</span>`;
      lucide.createIcons();
    }
  });

  // JSON Audit Export
  btnDownloadJson.addEventListener("click", () => {
    if (!lastAnalysisResult) return;
    const jsonStr = JSON.stringify(lastAnalysisResult, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `forensic_audit_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });

  // Benchmark Study Modal Handlers
  btnBenchmarkModal.addEventListener("click", async () => {
    benchmarkModal.classList.remove("hidden");
    try {
      const resp = await fetch("/api/benchmark");
      const data = await resp.json();
      modalRocImg.src = data.roc_curve;
      modalCmImg.src = data.confusion_matrix_chart;
    } catch (err) {
      console.error("Failed to load benchmark visuals:", err);
    }
  });

  btnCloseModal.addEventListener("click", () => {
    benchmarkModal.classList.add("hidden");
  });

  benchmarkModal.addEventListener("click", (e) => {
    if (e.target === benchmarkModal) {
      benchmarkModal.classList.add("hidden");
    }
  });
});
