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
from models.bst.pipeline.file_based_pipeline import FileBased_Pipeline, PipelineResult

# Import LSTM wrapper for dual analysis
from core.lstm_wrapper import LSTMWrapper, LSTMResult


# ---------------------------
# Helpers to format UI output
# ---------------------------

def _fmt_prediction(result: PipelineResult, threshold: float) -> str:
    conf_val = result.confidence if result.confidence is not None else None
    conf_str = f"{conf_val:.1%}" if conf_val is not None else "N/A"

    warn = ""
    if conf_val is not None and conf_val < threshold:
        warn = f" ⚠️ (below {threshold:.0%} threshold)"

    md = [
        "## 🏸 Stroke Classification Result",
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
            bar = "█" * bar_len + "░" * (20 - bar_len)
            md.append(f"{i}. **{pred['class']}** – {p:.1%}  `{bar}`")
    else:
        md.append("_No detailed predictions available_")

    return "\n".join(md)


def _fmt_lstm_prediction(lstm_result: LSTMResult, threshold: float) -> str:
    """Format LSTM analysis results similar to BST format"""
    if not lstm_result.success:
        return f"""
## 🧠 LSTM Analysis Result
❌ **Analysis Failed:** {lstm_result.error_message}
"""

    conf_str = f"{lstm_result.confidence:.1%}" if lstm_result.confidence else "N/A"

    warn = ""
    if lstm_result.confidence and lstm_result.confidence < threshold:
        warn = f" ⚠️ (below {threshold:.0%} threshold)"

    md = [
        "## 🧠 LSTM Analysis Result",
        f"**Performance Grade:** `{lstm_result.primary_prediction}`{warn}",
        f"**Confidence:** {conf_str}",
        f"**Processing Time:** {lstm_result.processing_time:.2f} seconds" if lstm_result.processing_time else "",
        "",
        "### Top 3 Predictions:",
    ]

    if lstm_result.predictions:
        for i, pred in enumerate(lstm_result.predictions[:3], 1):
            p = pred['confidence']
            bar_len = int(p * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            md.append(f"{i}. **{pred['shot_type']}** – {p:.1%}  `{bar}`")
    else:
        md.append("_No detailed predictions available_")

    return "\n".join(md)


def _lstm_confidence_html(lstm_result: LSTMResult) -> str:
    """Format LSTM confidence bars similar to BST format"""
    if not lstm_result.success or not lstm_result.predictions:
        return "<p>No LSTM confidence data available.</p>"

    rows = []
    for i, pred in enumerate(lstm_result.predictions[:3]):
        pct = pred['confidence'] * 100.0
        color = "#17a2b8" if i == 0 else "#6c757d"  # Different color from BST (blue vs green)
        rows.append(f"""
        <div style="margin: 10px 0;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-weight:bold;color:{color};">{pred['shot_type']}</span>
                <span style="color:{color};">{pct:.1f}%</span>
            </div>
            <div style="width:100%;background:#e9ecef;border-radius:4px;height:8px;">
                <div style="width:{pct:.1f}%;background:{color};height:8px;border-radius:4px;"></div>
            </div>
        </div>
        """)

    return f"""
    <div style="font-family:Arial, sans-serif;">
      <h3>🎯 LSTM Confidence Breakdown</h3>
      <div>{"".join(rows)}</div>
    </div>
    """


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
      <h3>📊 Confidence Breakdown</h3>
      <div>{"".join(rows)}</div>
    </div>
    """


def _fmt_technical(result: PipelineResult) -> str:
    pose_name = result.pose_file.name if result.pose_file else "N/A"
    csv_name = result.tracknet_csv.name if result.tracknet_csv else "N/A"
    vid_name = result.tracknet_video.name if result.tracknet_video else "N/A"

    md = f"""
## 📈 Technical Analysis

### Pipeline Status
- ✅ Video Validation
- ✅ MMPose Pose Detection
- ✅ TrackNetV3 Shuttlecock Tracking
- ✅ BST_8 Inference (JnB_bone)

### Generated Files
- **Pose Data:** {pose_name}
- **Shuttlecock CSV:** {csv_name}
- **Prediction Video:** {vid_name}

### Model
- **Architecture:** BST-8 (Badminton Stroke Transformer)
- **Pose Input:** joints + bones (COCO pairs), 2D
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
        self.lstm_wrapper: Optional[LSTMWrapper] = None
        self._log_lines: list[str] = []

    def _get_pipeline(self) -> FileBased_Pipeline:
        if self.pipeline is None:
            self.pipeline = FileBased_Pipeline()
        return self.pipeline

    def _get_lstm_wrapper(self) -> LSTMWrapper:
        if self.lstm_wrapper is None:
            self.lstm_wrapper = LSTMWrapper()
        return self.lstm_wrapper

    # This callback is passed down to the pipeline so we can surface status
    def _progress(self, message: str, progress: float) -> None:
        # Keep a small rolling log for the UI
        line = f"[{int(progress*100):>3}%] {message}"
        print(line, flush=True)
        self._log_lines.append(line)
        if len(self._log_lines) > 200:
            self._log_lines = self._log_lines[-200:]

    def _parse_court_corners(self, court_str: str):
        """Parse court corner string into list of tuples."""
        if not court_str or not court_str.strip():
            return None

        try:
            # Expected format: "x1,y1 x2,y2 x3,y3 x4,y4"
            pairs = court_str.strip().split()
            if len(pairs) != 4:
                return None

            corners = []
            for pair in pairs:
                x, y = pair.split(',')
                corners.append((int(x.strip()), int(y.strip())))

            return corners
        except Exception as e:
            print(f"Error parsing court corners: {e}")
            return None

    def _run(self, video_file: str, threshold: float, export_json: bool, court_corners_str: str):
        """Main handler for the Analyze button - now with dual BST + LSTM analysis."""
        self._log_lines = []
        if not video_file:
            return ("❌ No video uploaded.",
                    "<p>No confidence data.</p>",
                    "No technical details.",
                    "❌ No video uploaded.",
                    "<p>No LSTM confidence data.</p>",
                    None, None, None,
                    "")

        # Parse court corners
        court_corners = self._parse_court_corners(court_corners_str)

        # BST Analysis (ENABLED FOR BST-ONLY TESTING)
        self._progress("🤖 Starting BST Transformer analysis...", 0.1)
        pipe = self._get_pipeline()
        result = pipe.process_video(video_file, self._progress, court_corners)

        if not result.success:
            log_txt = "\n".join(self._log_lines + [f"[x] BST Error: {result.error_message}"])
            return (f"❌ BST Processing failed: {result.error_message}",
                    "<p>No confidence data.</p>",
                    "No technical details.",
                    "❌ BST failed, skipping LSTM.",
                    "<p>No LSTM confidence data.</p>",
                    None, None, None,
                    log_txt)

        # Build BST UI pieces
        pred_md = _fmt_prediction(result, threshold)
        conf_html = _confidence_html(result)
        tech_md = _fmt_technical(result)

        # LSTM Analysis (ENABLED FOR DUAL-METHOD APP)
        self._progress("🧠 Starting LSTM analysis...", 0.7)
        lstm_wrapper = self._get_lstm_wrapper()

        # Since LSTM wrapper is async, we need to handle it synchronously in Gradio
        import asyncio
        try:
            # Create or get event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run LSTM analysis
            lstm_result = loop.run_until_complete(lstm_wrapper.analyze_video(video_file))
        except Exception as e:
            self._progress(f"🚨 LSTM analysis error: {str(e)}", 0.9)
            # Create placeholder LSTM result
            lstm_result = LSTMResult(
                success=False,
                predictions=[],
                primary_prediction="Analysis Failed",
                confidence=0.0,
                performance_grade="N/A",
                grade_score=0.0,
                processing_time=0.0,
                error_message=str(e)
            )

        # Build LSTM UI pieces
        lstm_pred_md = _fmt_lstm_prediction(lstm_result, threshold)
        lstm_conf_html = _lstm_confidence_html(lstm_result)

        # Downloads (BST results available)
        json_path = _export_raw_json(result) if export_json else None
        video_path = str(result.tracknet_video) if result.tracknet_video else None

        # Combined visualization video
        combined_viz_path = str(result.combined_viz_video) if result.combined_viz_video else None

        log_txt = "\n".join(self._log_lines + ["[done] ✅ Dual analysis finished."])
        return (pred_md, conf_html, tech_md, lstm_pred_md, lstm_conf_html, combined_viz_path, json_path, video_path, log_txt)

    def build(self):
        css = """
        .gradio-container { font-family: Arial, sans-serif !important; }
        .result-box { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 14px; }
        .prediction-card {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            height: 100%;
        }
        .model-header {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid #667eea;
        }
        """

        with gr.Blocks(title="🏸 Badminton Stroke Classifier", css=css) as demo:
            gr.HTML(
                """<div style="text-align:center;padding:20px;border-radius:12px;
                background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <h1 style="margin:0 0 12px 0;font-size:32px;">🏸 Badminton Stroke Classifier</h1>
                <p style="margin:0;font-size:16px;opacity:0.95;">AI-Powered Dual-Method Stroke Analysis</p>
                </div>"""
            )

            with gr.Row():
                # Column 1: Video Upload & Settings
                with gr.Column(scale=1):
                    gr.Markdown("### 📹 Video Upload")
                    vid = gr.Video(label="Upload stroke clip (0.5–30s)", sources=["upload"], include_audio=False)

                    gr.Markdown("### ⚙️ Settings")
                    thr = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Confidence threshold")
                    want_json = gr.Checkbox(value=True, label="Export raw JSON summary")

                    with gr.Accordion("🏸 Court Boundaries (Optional)", open=False):
                        gr.Markdown("Enter 4 court corner coordinates as: `x1,y1 x2,y2 x3,y3 x4,y4`")
                        court_input = gr.Textbox(
                            label="Court corners",
                            placeholder="e.g., 100,200 800,200 800,600 100,600",
                            value=""
                        )

                    run_btn = gr.Button("🚀 Analyze Stroke", variant="primary", size="lg")

                    gr.Markdown("### 📁 Downloads")
                    out_json = gr.File(label="Raw JSON", interactive=False)
                    out_vid = gr.File(label="TrackNet predicted video", interactive=False)

                    with gr.Accordion("🧾 Processing Logs", open=False):
                        out_log = gr.Textbox(value="", lines=8, max_lines=15, show_copy_button=True)

                # Column 2: Visualization
                with gr.Column(scale=1):
                    gr.Markdown("### 🎥 Analysis Visualization")
                    out_viz = gr.Video(label="Pose + Shuttlecock Tracking", interactive=False)
                    gr.Markdown("""
                    <div style="text-align:center;padding:12px;background:#e8f5e9;border-radius:8px;margin-top:10px;">
                    <small>📊 <b>Combined view:</b> Player skeletons (green/magenta) + shuttlecock tracking (red) overlaid on your video.
                    Court detection coming soon!</small>
                    </div>
                    """)

                # Column 3: Side-by-Side Results (BST & LSTM)
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Prediction Results")

                    with gr.Row(equal_height=True):
                        # LSTM Results (Left)
                        with gr.Column(scale=1):
                            gr.HTML('<div class="model-header">🧠 LSTM Analysis</div>')
                            out_lstm_pred = gr.Markdown(
                                value="Awaiting analysis...",
                                elem_classes=["result-box"]
                            )
                            out_lstm_conf = gr.HTML(value="<p style='text-align:center;color:#999;'>Confidence bars will appear here</p>")

                        # BST Results (Right)
                        with gr.Column(scale=1):
                            gr.HTML('<div class="model-header">🤖 BST Transformer</div>')
                            out_pred = gr.Markdown(
                                value="Awaiting analysis...",
                                elem_classes=["result-box"]
                            )
                            out_conf = gr.HTML(value="<p style='text-align:center;color:#999;'>Confidence bars will appear here</p>")

                    # Technical Details (Full Width)
                    with gr.Accordion("📈 Technical Details", open=False):
                        out_tech = gr.Markdown(value="Technical details will appear here.")

            # Wire events
            run_btn.click(
                fn=self._run,
                inputs=[vid, thr, want_json, court_input],
                outputs=[out_pred, out_conf, out_tech, out_lstm_pred, out_lstm_conf, out_viz, out_json, out_vid, out_log],
            )

            gr.Markdown("""
---
**Pipeline:** MMPose → TrackNetV3 → BST-8 (35 stroke types) + YOLO Pose → LSTM (performance grading)
**Note:** GPU/CUDA compatibility issues will automatically fall back to CPU processing.
""")

        return demo


# -------------
# Entrypoint
# -------------
if __name__ == "__main__":
    app = GradioApp()
    ui = app.build().queue()  # keep simple for broad version compatibility
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True,
    )
