#!/usr/bin/env python3
"""
Gradio UI for Badminton Stroke Classification
- Uses the file-based pipeline that runs: MMPose -> TrackNetV3 -> BST_8 (JnB_bone)
- Shows prediction, confidence bars, technical details, and download files
"""

import gradio as gr
import json
import time
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Import our pipeline from the pipeline package
from pipeline.file_based_pipeline import FileBased_Pipeline, PipelineResult


# ---------------------------
# Helpers to format UI output
# ---------------------------

def _fmt_prediction(result: PipelineResult, threshold: float) -> str:
    conf_val = result.confidence if result.confidence is not None else None
    conf_str = f"{conf_val:.1%}" if conf_val is not None else "N/A"

    warn = ""
    if conf_val is not None and conf_val < threshold:
        warn = f" (below {threshold:.0%} threshold)"

    md = [
        "## Stroke Classification Result",
        f"**Predicted Stroke:** `{result.stroke_prediction}`{warn}",
        f"**Confidence:** {conf_str}",
        f"**Processing Time:** {result.processing_time:.2f} seconds" if result.processing_time else "",
        "",
        "### Top 3 Predictions:",
    ]

    if result.top3_predictions:
        for i, pred in enumerate(result.top3_predictions, 1):
            p = float(pred["confidence"])
            bar_len = int(p * 20)
            bar = "" * bar_len + "" * (20 - bar_len)
            md.append(f"{i}. **{pred['class']}** – {p:.1%} `{bar}`")
    else:
        md.append("_No detailed predictions available_")

    return "\n".join(md)


def _confidence_html(result: PipelineResult) -> str:
    if not result.top3_predictions:
        return "<p>No confidence data available.</p>"

    rows = []
    for i, pred in enumerate(result.top3_predictions):
        pct = float(pred["confidence"]) * 100.0
        color = "#28a745" if i == 0 else "#6c757d"
        rows.append(f"""
        <div style="margin: 10px 0;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-weight:bold;color:{color};">{pred['class']}</span>
                <span style="color:{color};">{pct:.1f}%</span>
            </div>
            <div style="width:100%;background:#e9ecef;border-radius:4px;height:8px;">
                <div style="width:{pct}%;background:{color};height:8px;border-radius:4px;"></div>
            </div>
        </div>
        """)

    return f"""
    <div style="font-family:Arial, sans-serif;">
      <h3> Confidence Breakdown</h3>
      <div>{"".join(rows)}</div>
    </div>
    """


def _fmt_technical(result: PipelineResult) -> str:
    pose_name = result.pose_file.name if result.pose_file else "N/A"
    csv_name = result.tracknet_csv.name if result.tracknet_csv else "N/A"
    vid_name = result.tracknet_video.name if result.tracknet_video else "N/A"

    md = f"""
## Technical Analysis

### Pipeline Status
- Video Validation
- MMPose Pose Detection
- TrackNetV3 Shuttlecock Tracking
- BST_8 Inference (JnB_bone)

### Generated Files
- **Pose Data:** {pose_name}
- **Shuttlecock CSV:** {csv_name}
- **Prediction Video:** {vid_name}

### Model
- **Architecture:**BST-8 (Badminton Stroke Transformer)
- **Pose Input:**joints + bones (COCO pairs), 2D
- **Sequence Length:** 30 frames (padded/truncated)
- **Classes:** 35 stroke types
"""
    return md


def _export_raw_json(result: PipelineResult) -> Optional[str]:
    """Create a JSON bundle for download (prediction + file paths summary)."""
    try:
        payload: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": {
                "stroke": result.stroke_prediction,
                "confidence": float(result.confidence) if result.confidence is not None else None,
                "top3_predictions": result.top3_predictions or [],
            },
            "processing_time": float(result.processing_time) if result.processing_time is not None else None,
            "files": {
                "pose_file": str(result.pose_file) if result.pose_file else None,
                "tracknet_csv": str(result.tracknet_csv) if result.tracknet_csv else None,
                "tracknet_video": str(result.tracknet_video) if result.tracknet_video else None,
            },
        }

        # Try to inline a preview of the first ~50 shuttle rows (handy for debugging)
        if result.tracknet_csv and Path(result.tracknet_csv).exists():
            try:
                df = pd.read_csv(result.tracknet_csv)
                payload["tracknet_head"] = df.head(50).to_dict(orient="records")
            except Exception:
                pass

        out = Path(result.temp_dir or ".") / "bst_result.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(out)
    except Exception:
        return None


# ---------------------------
# Gradio App
# ---------------------------

class GradioApp:
    def __init__(self):
        self.pipeline: Optional[FileBased_Pipeline] = None
        self._log_lines: list[str] = []

    def _get_pipeline(self) -> FileBased_Pipeline:
        if self.pipeline is None:
            self.pipeline = FileBased_Pipeline()
        return self.pipeline

    # This callback is passed down to the pipeline so we can surface status
    def _progress(self, message: str, progress: float) -> None:
        # Keep a small rolling log for the UI
        line = f"[{int(progress*100):>3}%] {message}"
        print(line, flush=True)
        self._log_lines.append(line)
        if len(self._log_lines) > 200:
            self._log_lines = self._log_lines[-200:]

    def _run(self, video_file: str, threshold: float, export_json: bool):
        """Main handler for the Analyze button."""
        self._log_lines = []
        if not video_file:
            return (" No video uploaded.",
                    "<p>No confidence data.</p>",
                    "No technical details.",
                    None, None,
                    "")

        pipe = self._get_pipeline()
        result = pipe.process_video(video_file, self._progress)

        if not result.success:
            log_txt = "\n".join(self._log_lines + [f"[x] Error: {result.error_message}"])
            return (f" Processing failed: {result.error_message}",
                    "<p>No confidence data.</p>",
                    "No technical details.",
                    None, None,
                    log_txt)

        # Build UI pieces
        pred_md = _fmt_prediction(result, threshold)
        conf_html = _confidence_html(result)
        tech_md = _fmt_technical(result)

        # Downloads
        json_path = _export_raw_json(result) if export_json else None
        video_path = str(result.tracknet_video) if result.tracknet_video else None

        log_txt = "\n".join(self._log_lines + ["[done] Finished."])
        return (pred_md, conf_html, tech_md, json_path, video_path, log_txt)

    def build(self):
        css = """
        .gradio-container { font-family: Arial, sans-serif !important; }
        .result-box { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 14px; }
        """

        with gr.Blocks(title=" Badminton Stroke Classifier", css=css) as demo:
            gr.HTML(
                """<div style="text-align:center;padding:18px;border-radius:10px;
                background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);color:white;">
                <h1> Badminton Stroke Classifier</h1>
                <p>Upload a video — we’ll run MMPose TrackNetV3 BST-8 (JnB_bone) and show the prediction.</p>
                </div>"""
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Video")
                    vid = gr.Video(label="Upload stroke clip (0.5–30s)", sources=["upload"], include_audio=False)

                    gr.Markdown("### Settings")
                    thr = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Confidence threshold")
                    want_json = gr.Checkbox(value=True, label="Export raw JSON summary")

                    run_btn = gr.Button(" Analyze Stroke", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### Results")
                    out_pred = gr.Markdown(value="Upload a video and click **Analyze**.", elem_classes=["result-box"])
                    out_conf = gr.HTML(value="<p>Confidence bars will appear here.</p>")
                    out_tech = gr.Markdown(value="Technical details will appear here.")

                    gr.Markdown("### Downloads")
                    out_json = gr.File(label="Raw JSON", interactive=False)
                    out_vid = gr.File(label="TrackNet predicted video", interactive=False)

                    gr.Markdown("### Logs")
                    out_log = gr.Textbox(value="", lines=10, max_lines=20, show_copy_button=True)

            # Wire events (no special queue args to keep compat with older Gradio)
            run_btn.click(
                fn=self._run,
                inputs=[vid, thr, want_json],
                outputs=[out_pred, out_conf, out_tech, out_json, out_vid, out_log],
            )

            gr.Markdown("""
**Notes**
- If you see a message about GPU/CUDA compatibility, the pipeline will fall back to CPU safely.
- Files are kept in a temporary folder long enough for you to download them.
""")

        return demo


# -------------
# Entrypoint
# -------------
if __name__ == "__main__":
    app = GradioApp()
    ui = app.build().queue() # keep simple for broad version compatibility
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True,
    )
