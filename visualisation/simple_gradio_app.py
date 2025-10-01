#!/usr/bin/env python3
"""
ABOUTME: Simple Gradio app for BST badminton stroke classification
ABOUTME: Upload video -> preprocess -> BST inference -> results
"""

import gradio as gr
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

# Stroke type mapping
STROKE_TYPES = [
    'Top_backcourt_BH_defensive',
    'Top_backcourt_BH_drive',
    'Top_backcourt_BH_lob',
    'Top_backcourt_BH_rush',
    'Top_backcourt_FH_defensive',
    'Top_backcourt_FH_drive',
    'Top_backcourt_FH_lob',
    'Top_backcourt_FH_rush',
    'Top_backcourt_FH_smash',
    'Top_midcourt_BH_drive',
    'Top_midcourt_BH_push',
    'Top_midcourt_FH_drive',
    'Top_midcourt_FH_push',
    'Top_frontcourt_BH_net_shot',
    'Top_frontcourt_FH_net_shot',
    'Bottom_backcourt_BH_defensive',
    'Bottom_backcourt_BH_lob',
    'Bottom_backcourt_FH_defensive',
    'Bottom_backcourt_FH_lob',
    'Bottom_midcourt_BH_drive',
    'Bottom_midcourt_FH_drive',
    'Bottom_cross-court_net_shot',
    'Bottom_straight_net_shot'
]


def process_video(video_path, progress=gr.Progress()):
    """Process a badminton video and classify the stroke type."""

    if video_path is None:
        return "Please upload a video file.", None

    try:
        # Create temporary directory for processing
        temp_dir = Path(tempfile.mkdtemp(prefix="bst_simple_"))
        npy_dir = temp_dir / "npy"
        npy_dir.mkdir(exist_ok=True)

        progress(0.1, desc="Starting preprocessing...")

        # Step 1: Run preprocessing (TrackNet + MMPose + collation)
        progress(0.2, desc="Running TrackNet and MMPose...")

        process_script = Path("src/process_single_video.py")
        cmd = [
            sys.executable,
            str(process_script),
            str(video_path),
            "-o", str(npy_dir),
            "--seq-len", "100",
            "--keep-intermediates"
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            error_msg = f"Preprocessing failed:\n{result.stderr}"
            print(error_msg)
            return error_msg, None

        print("Preprocessing output:")
        print(result.stdout)

        progress(0.6, desc="Preprocessing complete, running BST inference...")

        # Step 2: Prepare files for BST inference
        # run_bst_on_triplet.py expects files named: {base}_joints.npy, {base}_pos.npy, {base}_shuttle.npy
        intermediates_dir = npy_dir / "intermediates"
        video_stem = Path(video_path).stem

        # Copy/rename intermediate files to match expected pattern
        base_name = npy_dir / video_stem
        joints_src = intermediates_dir / f"{video_stem}_joints.npy"
        pos_src = intermediates_dir / f"{video_stem}_pos.npy"
        shuttle_src = intermediates_dir / f"{video_stem}_shuttle.npy"

        joints_dst = Path(str(base_name) + "_joints.npy")
        pos_dst = Path(str(base_name) + "_pos.npy")
        shuttle_dst = Path(str(base_name) + "_shuttle.npy")

        if not joints_src.exists():
            return f"Error: Expected intermediate file not found: {joints_src}", None

        shutil.copy2(joints_src, joints_dst)
        shutil.copy2(pos_src, pos_dst)
        shutil.copy2(shuttle_src, shuttle_dst)

        # Run BST inference
        bst_script = Path("models/bst/pipeline/run_bst_on_triplet.py")
        bst_cmd = [
            sys.executable,
            str(bst_script),
            str(base_name),
            "--weights", "weights/bst_8_JnB_bone_bottom_frontier_6class.pt",
            "--device", "cuda"
        ]

        print(f"Running BST: {' '.join(bst_cmd)}")
        bst_result = subprocess.run(bst_cmd, capture_output=True, text=True, timeout=60)

        if bst_result.returncode != 0:
            error_msg = f"BST inference failed:\n{bst_result.stderr}"
            print(error_msg)
            return error_msg, None

        print("BST output:")
        print(bst_result.stdout)

        progress(0.9, desc="Processing results...")

        # Parse the output to get predicted class and confidence
        output_lines = bst_result.stdout.strip().split('\n')
        predicted_class = None
        predicted_conf = None
        top3_results = []

        for i, line in enumerate(output_lines):
            if line.startswith("Class:"):
                predicted_class = line.split(":")[-1].strip()
            elif line.startswith("Confidence:"):
                predicted_conf = line.split(":")[-1].strip()
            elif line.startswith("Top-3:"):
                # Parse the top-3 results
                for j in range(i + 1, min(i + 4, len(output_lines))):
                    if output_lines[j].strip() and output_lines[j][0].isdigit():
                        top3_results.append(output_lines[j])

        if predicted_class is None:
            return f"Could not parse BST output:\n{bst_result.stdout}", None

        # Format top-3 results nicely
        top3_display = "\n".join(top3_results) if top3_results else "N/A"

        result_text = f"""
# Stroke Classification Result

**Predicted Stroke Type:** {predicted_class}

**Confidence:** {predicted_conf}

## Top-3 Predictions:
```
{top3_display}
```

---

### Processing Summary:
- Video processed successfully ✓
- TrackNet shuttlecock detection: ✓
- MMPose player pose estimation: ✓
- BST Transformer inference: ✓

Temporary files saved to: `{temp_dir}`
"""

        progress(1.0, desc="Complete!")

        return result_text, str(video_path)

    except subprocess.TimeoutExpired:
        return "Error: Processing timed out (>10 minutes)", None
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, None


# Create Gradio interface
with gr.Blocks(title="BST Badminton Stroke Classifier") as demo:
    gr.Markdown("# 🏸 BST Badminton Stroke Classifier")
    gr.Markdown("Upload a badminton video to classify the stroke type using BST Transformer")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Badminton Video")
            submit_btn = gr.Button("Classify Stroke", variant="primary")

        with gr.Column():
            result_output = gr.Markdown(label="Results")
            video_output = gr.Video(label="Input Video")

    submit_btn.click(
        fn=process_video,
        inputs=[video_input],
        outputs=[result_output, video_output]
    )

    gr.Markdown("""
    ## About
    This app uses:
    - **TrackNetV3** for shuttlecock detection
    - **MMPose (RTMPose)** for player pose estimation
    - **BST Transformer** for stroke classification

    Supported stroke types: 23 different badminton strokes (see STROKE_TYPES list)
    """)


if __name__ == "__main__":
    demo.launch(share=True)
