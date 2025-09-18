#!/usr/bin/env python3
"""
File-based video processing pipeline: MMPose → TrackNetV3 → BST (single video)
Runs single-video MMPose, TrackNet, stages a triplet, then calls run_bst_on_triplet.py.
"""

import re
import os
import sys
import json
import time
import shutil
import subprocess
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

# Ensure local imports
sys.path.append(str(Path(__file__).parent))

SEQ_LEN = 30
N_PLAYERS = 2


@dataclass
class PipelineResult:
    success: bool
    stroke_prediction: Optional[str] = None
    confidence: Optional[float] = None
    top3_predictions: Optional[List[Dict]] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None

    # artifacts
    pose_file: Optional[Path] = None
    pos_file: Optional[Path] = None
    tracknet_csv: Optional[Path] = None
    tracknet_video: Optional[Path] = None
    bst_root: Optional[Path] = None
    temp_dir: Optional[Path] = None


def _run_and_stream(
    cmd: List[str],
    workdir: Path,
    progress_cb: Optional[Callable[[str, float], None]] = None,
    start_msg: str = "",
    start_progress: float = 0.0,
    done_msg: str = "",
    done_progress: float = 0.0,
    timeout_sec: int = 1200,
) -> Tuple[int, str]:
    if progress_cb and start_msg:
        progress_cb(start_msg, start_progress)

    # Unbuffer child Python
    if cmd and cmd[0] == "python" and "-u" not in cmd:
        cmd = cmd[:1] + ["-u"] + cmd[1:]

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    proc = subprocess.Popen(
        cmd, cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True, env=env
    )

    out_lines: List[str] = []
    t0 = time.time()
    try:
        for line in proc.stdout or []:
            out_lines.append(line)
            ll = line.strip().lower()
            if progress_cb and any(k in ll for k in ("error", "warning", "epoch", "iter", "inference", "loaded", "saving")):
                progress_cb(line.strip(), min(done_progress, start_progress + 0.05))
            if time.time() - t0 > timeout_sec:
                proc.kill()
                out_lines.append(f"[Timeout after {timeout_sec}s]")
                break
        rc = proc.wait(timeout=5)
    except Exception as e:
        rc = 1
        out_lines.append(f"[Subprocess error] {e}")

    if progress_cb and done_msg:
        progress_cb(done_msg, done_progress)

    return rc, "".join(out_lines)


def _stage_triplet_for_bst(
    video_stem: str,
    temp_dir: Path,
    bst_data_dir: Path,
    seq_len: int = SEQ_LEN,
) -> Path:
    """
    Create a bst_format directory and stage the three files expected by run_bst_on_triplet.py:
      <stem>_joints.npy, <stem>_pos.npy, <stem>_shuttle.npy
    Returns the directory containing the staged files.
    """
    staged_dir = temp_dir / "bst_format" / "test" / "Top_殺球"
    staged_dir.mkdir(parents=True, exist_ok=True)

    src_joints = bst_data_dir / "poses.npy"
    src_shuttle = bst_data_dir / "shuttlecock.npy"
    src_pos = bst_data_dir / "pos.npy"

    if not src_joints.exists():
        raise FileNotFoundError(f"poses.npy not found at {src_joints}")
    if not src_shuttle.exists():
        raise FileNotFoundError(f"shuttlecock.npy not found at {src_shuttle}")
    if not src_pos.exists():
        # zeros fallback
        np.save(src_pos, np.zeros((seq_len, 2, 2), dtype=float))

    np.save(staged_dir / f"{video_stem}_joints.npy", np.load(src_joints))
    np.save(staged_dir / f"{video_stem}_shuttle.npy", np.load(src_shuttle))
    np.save(staged_dir / f"{video_stem}_pos.npy", np.load(src_pos))

    return staged_dir


class FileBased_Pipeline:
    def __init__(
        self,
        bst_weight_path: str = "weights/bst_model.pt",
        tracknet_model_path: str = "weights/tracknet_model.pt",
    ):
        self.bst_weight_path = Path(bst_weight_path)
        self.tracknet_model_path = Path(tracknet_model_path)

        # TrackNet script (we'll use our own implementation)
        self.tracknet_script = Path("tracknet/predict.py")

        self._verify_components()
        print("File-based Pipeline initialized")
        print(f"  BST weights: {self.bst_weight_path}")
        print(f"  TrackNet weights: {self.tracknet_model_path}")

    def _verify_components(self):
        missing = []
        if not self.bst_weight_path.exists():
            missing.append(f"BST weights: {self.bst_weight_path}")
        if not self.tracknet_model_path.exists():
            missing.append(f"TrackNet weights: {self.tracknet_model_path}")
        if not self.tracknet_script.exists():
            missing.append(f"TrackNet script: {self.tracknet_script}")
        if missing:
            raise FileNotFoundError(f"Missing components: {missing}")

    # ---------- Utils ----------

    def validate_video(self, video_path: Path) -> Tuple[bool, str]:
        if not video_path.exists():
            return False, f"Video file not found: {video_path}"
        if video_path.suffix.lower() not in [".mp4", ".avi", ".mov", ".mkv"]:
            return False, f"Unsupported video format: {video_path.suffix}"

        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
                capture_output=True, text=True, timeout=30
            )
            if r.returncode == 0 and r.stdout:
                info = json.loads(r.stdout)
                duration = float(info["format"]["duration"])
                if duration < 0.5:
                    return False, f"Video too short: {duration:.1f}s (min 0.5s)"
                if duration > 30:
                    return False, f"Video too long: {duration:.1f}s (max 30s)"
                return True, f"Video valid: {duration:.1f}s"
        except Exception:
            pass

        size_mb = video_path.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            return False, f"Video too large: {size_mb:.1f}MB (max 100MB)"
        return True, f"Video appears valid: {size_mb:.1f}MB"

    # ---------- MMPose (single video) ----------

    def _mmpose_single_video(
        self,
        video_path: Path,
        output_dir: Path,
        progress_cb: Optional[Callable[[str, float], None]] = None,
        target_frames: int = SEQ_LEN,
    ) -> Tuple[bool, Optional[Path], Optional[Path], str]:
        try:
            from mmpose.apis import MMPoseInferencer
        except Exception as e:
            return False, None, None, f"MMPose not available: {e}"

        device = os.environ.get("BST_DEVICE", "cpu")  # CPU by default
        output_dir.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb(f"Running MMPose on {video_path.name} (device={device})…", 0.10)

        try:
            infer = MMPoseInferencer("human", device=device)
        except Exception as e:
            return False, None, None, f"MMPoseInferencer init failed: {e}"

        key_seq: List[np.ndarray] = []
        pos_seq: List[np.ndarray] = []
        try:
            for i, res in enumerate(infer(str(video_path), show=False)):
                preds = res.get("predictions", [])
                if not preds:
                    key_seq.append(np.zeros((2, 17, 2), dtype=float))
                    pos_seq.append(np.zeros((2, 2), dtype=float))
                else:
                    persons = preds[0]  # batch=1
                    # sort by bbox center y (top player first)
                    persons_sorted = sorted(
                        persons,
                        key=lambda p: ((p.get("bbox")[0][1] + p.get("bbox")[0][3]) / 2.0) if p.get("bbox") else 1e9
                    )
                    two = persons_sorted[:2]
                    while len(two) < 2:
                        two.append({"keypoints": np.zeros((17, 2)), "bbox": [[0, 0, 0, 0]]})

                    kp = np.zeros((2, 17, 2), dtype=float)
                    pp = np.zeros((2, 2), dtype=float)
                    for kidx, p in enumerate(two):
                        k = np.array(p.get("keypoints", np.zeros((17, 2))), dtype=float)
                        if k.ndim != 2 or k.shape[0] < 17:
                            k = np.zeros((17, 2), dtype=float)
                        kp[kidx, :, :] = k[:17, :2]

                        bb = p.get("bbox")
                        if bb is not None:
                            bb = bb[0] if isinstance(bb[0], (list, tuple, np.ndarray)) else bb
                            if len(bb) >= 4:
                                pp[kidx, 0] = (bb[0] + bb[2]) / 2.0
                                pp[kidx, 1] = (bb[1] + bb[3]) / 2.0

                    key_seq.append(kp)
                    pos_seq.append(pp)

                if progress_cb and (i % 5 == 0 or i + 1 == target_frames):
                    frac = min(1.0, (i + 1) / max(1, target_frames))
                    progress_cb(f"MMPose frames processed: {i+1}", 0.10 + 0.20 * frac)

                if len(key_seq) >= target_frames:
                    break
        except Exception as e:
            return False, None, None, f"MMPose inference failed: {e}"

        if len(key_seq) == 0:
            return False, None, None, "MMPose produced no frames."

        keys = np.stack(key_seq)  # (T, 2, 17, 2)
        poss = np.stack(pos_seq)  # (T, 2, 2)
        T = keys.shape[0]
        if T < target_frames:
            pad_k = np.repeat(keys[-1:], target_frames - T, axis=0)
            pad_p = np.repeat(poss[-1:], target_frames - T, axis=0)
            keys = np.concatenate([keys, pad_k], axis=0)
            poss = np.concatenate([poss, pad_p], axis=0)

        pose_file = output_dir / "poses.npy"
        pos_file = output_dir / "pos.npy"
        try:
            np.save(pose_file, keys)
            np.save(pos_file, poss)
        except Exception as e:
            return False, None, None, f"Failed to save pose/pos arrays: {e}"

        if progress_cb:
            progress_cb("MMPose detection completed", 0.30)

        return True, pose_file, pos_file, f"MMPose OK: {keys.shape} saved"

    # ---------- TrackNet ----------

    def run_tracknet_detection(
        self,
        video_path: Path,
        temp_dir: Path,
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[bool, Optional[Path], Optional[Path], str]:
        tracknet_dir = temp_dir / "tracknet_output"
        tracknet_dir.mkdir(exist_ok=True)

        cmd = [
            "python", str(self.tracknet_script),
            "--video_file", str(video_path),
            "--model_file", str(self.tracknet_model_path),
            "--save_dir", str(tracknet_dir),
        ]
        print("TrackNet cmd:", " ".join(cmd))
        rc, out = _run_and_stream(
            cmd, Path.cwd(), progress_cb,
            "Running TrackNetV3 shuttlecock tracking...", 0.40,
            "TrackNetV3 tracking completed", 0.60, timeout_sec=1200
        )
        if rc != 0:
            return False, None, None, f"TrackNet failed (rc={rc}). Last logs:\n{out[-1200:]}"

        video_name = video_path.stem
        csv_file = tracknet_dir / f"{video_name}_ball.csv"
        pred_video = tracknet_dir / f"{video_name}_pred.{video_path.suffix[1:]}"
        if not csv_file.exists():
            return False, None, None, f"TrackNet CSV not found: {csv_file}"
        return True, csv_file, (pred_video if pred_video.exists() else None), "TrackNet OK"

    # ---------- BST helpers ----------

    def _convert_tracknet_to_bst_format(
        self, tracknet_csv: Path, output_dir: Path, target_frames: int = SEQ_LEN
    ) -> Path:
        """Read TrackNet CSV and save shuttlecock.npy in BST data dir."""
        try:
            df = pd.read_csv(tracknet_csv).drop_duplicates("Frame")
            xy = []
            for _, r in df.iterrows():
                vis = int(r.get("Visibility", 1))
                if vis == 1:
                    xy.append([float(r["X"]), float(r["Y"])])
                else:
                    xy.append([0.0, 0.0])
            arr = np.array(xy, dtype=float).reshape(-1, 2)
        except Exception:
            arr = np.zeros((1, 2), dtype=float)

        if len(arr) < target_frames:
            pad = np.repeat(arr[-1:], target_frames - len(arr), axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        elif len(arr) > target_frames:
            start = max(0, (len(arr) - target_frames) // 2)
            arr = arr[start:start + target_frames]

        out_path = output_dir / "shuttlecock.npy"
        np.save(out_path, arr)
        return out_path

    # ---------- BST (calls the proven script) ----------

    def run_bst_inference(
        self,
        pose_file: Path,
        pos_file: Path,
        tracknet_csv: Path,
        temp_dir: Path,
        progress_cb: Optional[Callable[[str, float], None]] = None,
        video_stem: Optional[str] = None,
    ) -> Tuple[bool, Dict, str, Optional[Path]]:
        """
        Returns (ok, bst_result_dict, message, bst_root_dir)
        in the exact order your Gradio code currently unpacks.
        """
        try:
            if progress_cb:
                progress_cb("Running BST stroke classification...", 0.80)

            # Build BST data dir (where we write poses.npy / pos.npy / shuttlecock.npy)
            bst_data_dir = temp_dir / "bst_data"
            bst_data_dir.mkdir(exist_ok=True)

            # Copy pose/pos to canonical names
            pose_dest = bst_data_dir / "poses.npy"
            shutil.copy2(pose_file, pose_dest)

            pos_dest = bst_data_dir / "pos.npy"
            shutil.copy2(pos_file, pos_dest)

            # Convert TrackNet CSV -> shuttlecock.npy (30 frames)
            self._convert_tracknet_to_bst_format(tracknet_csv, bst_data_dir, target_frames=SEQ_LEN)

            # Decide a stable key for files
            stem = video_stem if video_stem else pose_dest.stem

            # Stage the triplet for the proven script
            staged_dir = _stage_triplet_for_bst(stem, temp_dir=temp_dir, bst_data_dir=bst_data_dir, seq_len=SEQ_LEN)

            # Call the known-good script as a subprocess (CPU; your CUDA build mismatched)
            triplet_prefix = str(staged_dir / stem)
            cmd = [
                "python", "-u", "pipeline/run_bst_on_triplet.py",
                triplet_prefix,
                "--weights", str(self.bst_weight_path),
                "--device", "cpu",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()), timeout=600)
            if proc.returncode != 0:
                tail = proc.stderr.strip() or proc.stdout.strip()
                return False, {}, f"BST script failed (rc={proc.returncode}).\n{tail}", staged_dir

            out = proc.stdout

            # Parse the script’s stdout
            cls_match = re.search(r"^Class:\s*(.+)$", out, re.MULTILINE)
            conf_match = re.search(r"^Confidence:\s*([0-9]*\.?[0-9]+)", out, re.MULTILINE)
            top3_lines = re.findall(r"^\s*\d+\.\s+(.+?)\s+([0-9]*\.?[0-9]+)%", out, re.MULTILINE)

            if not cls_match or not conf_match:
                return False, {}, f"BST script ran but output could not be parsed:\n{out}", staged_dir

            pred_class = cls_match.group(1).strip()
            pred_conf = float(conf_match.group(1))

            top3 = []
            for cls, pct in top3_lines[:3]:
                cls = re.sub(r"\s{2,}", " ", cls).strip()
                try:
                    top3.append({"class": cls, "confidence": float(pct) / 100.0})
                except Exception:
                    pass

            result_dict = {
                "prediction": pred_class,
                "confidence": pred_conf,
                "top3_predictions": top3 if top3 else None,
            }

            if progress_cb:
                progress_cb("BST inference completed", 1.00)

            return True, result_dict, "BST inference completed successfully", staged_dir

        except subprocess.TimeoutExpired:
            return False, {}, "BST inference timed out", None
        except Exception as e:
            return False, {}, f"BST inference error: {e}", None

    # ---------- Orchestrate ----------

    def process_video(self, video_path: str, progress_callback: Optional[Callable[[str, float], None]] = None) -> PipelineResult:
        t0 = time.time()
        video_path = Path(video_path)
        temp_dir = Path(tempfile.mkdtemp(prefix="bst_pipeline_"))

        try:
            if progress_callback:
                progress_callback("Validating video...", 0.00)
            ok, msg = self.validate_video(video_path)
            if not ok:
                return PipelineResult(False, error_message=msg, temp_dir=temp_dir)

            # MMPose
            poses_dir = temp_dir / "poses"
            ok, pose_file, pos_file, msg = self._mmpose_single_video(video_path, poses_dir, progress_callback, target_frames=SEQ_LEN)
            if not ok or not pose_file or not pos_file:
                return PipelineResult(False, error_message=msg, temp_dir=temp_dir)

            # TrackNet
            ok, tracknet_csv, tracknet_video, msg = self.run_tracknet_detection(video_path, temp_dir, progress_callback)
            if not ok or not tracknet_csv:
                return PipelineResult(False, error_message=msg, temp_dir=temp_dir)

            # BST (ORDER: ok, bst_result, msg, bst_root)  ← matches your Gradio unpack
            ok, bst_result, msg, bst_root = self.run_bst_inference(
                pose_file, pos_file, tracknet_csv, temp_dir,
                video_stem=video_path.stem, progress_cb=progress_callback
            )
            if not ok:
                return PipelineResult(False, error_message=msg, temp_dir=temp_dir)

            t1 = time.time()
            return PipelineResult(
                success=True,
                stroke_prediction=bst_result.get("prediction"),
                confidence=bst_result.get("confidence"),
                top3_predictions=bst_result.get("top3_predictions"),
                processing_time=t1 - t0,
                pose_file=pose_file,
                pos_file=pos_file,
                tracknet_csv=tracknet_csv,
                tracknet_video=tracknet_video,
                bst_root=bst_root,
                temp_dir=temp_dir,
            )
        except Exception as e:
            return PipelineResult(False, error_message=f"Pipeline error: {e}", temp_dir=temp_dir)

    def cleanup(self, result: PipelineResult):
        if result.temp_dir and result.temp_dir.exists():
            shutil.rmtree(result.temp_dir, ignore_errors=True)
