# ABOUTME: Ball tracking prediction script for badminton videos using TrackNetV2
# ABOUTME: Processes video frames and outputs ball locations with visualization

import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import os
import sys
from collections import defaultdict

from tracknet.model import TrackNetV2
from tracknet.utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='TrackNet Ball Tracking')

    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--output_path', type=str, default='./output.mp4',
                        help='Path for output video')
    parser.add_argument('--csv_path', type=str, default='./predictions.csv',
                        help='Path for CSV output with predictions')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--height', type=int, default=288,
                        help='Input height for model')
    parser.add_argument('--width', type=int, default=512,
                        help='Input width for model')
    parser.add_argument('--sequence_length', type=int, default=3,
                        help='Number of frames in sequence')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for ball detection')

    return parser.parse_args()

def load_model(model_path, device):
    """Load the trained TrackNet model"""
    model = TrackNetV2(input_channels=9, out_channels=3)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

def preprocess_frames(frames, height, width):
    """Preprocess frames for model input"""
    processed_frames = []

    for frame in frames:
        # Resize frame
        resized = cv2.resize(frame, (width, height))
        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        processed_frames.append(normalized)

    return np.array(processed_frames)

def postprocess_predictions(predictions, original_height, original_width, model_height, model_width, threshold=0.5):
    """Convert model predictions to ball coordinates"""
    results = []

    for pred in predictions:
        # Get the heatmap (assuming channel 0 is ball heatmap)
        heatmap = pred[0]

        # Find the maximum value and its location
        max_val = np.max(heatmap)

        if max_val > threshold:
            max_loc = np.unravel_index(np.argmax(heatmap), heatmap.shape)

            # Convert to original video coordinates
            y = int(max_loc[0] * original_height / model_height)
            x = int(max_loc[1] * original_width / model_width)

            results.append({
                'visibility': 1,
                'x': x,
                'y': y,
                'confidence': float(max_val)
            })
        else:
            results.append({
                'visibility': 0,
                'x': -1,
                'y': -1,
                'confidence': float(max_val)
            })

    return results

def create_frame_sequences(frames, sequence_length=3):
    """Create sequences of frames for model input"""
    sequences = []

    for i in range(len(frames) - sequence_length + 1):
        sequence = frames[i:i + sequence_length]
        # Stack frames channel-wise (3 frames * 3 channels = 9 channels)
        stacked = np.concatenate(sequence, axis=2)
        # Transpose to CHW format
        stacked = np.transpose(stacked, (2, 0, 1))
        sequences.append(stacked)

    return np.array(sequences)

def main():
    args = parse_args()

    print(f"Loading video from: {args.video_path}")
    print(f"Using model: {args.model_path}")
    print(f"Device: {args.device}")

    # Load model
    model = load_model(args.model_path, args.device)
    print("Model loaded successfully")

    # Open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {args.video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {original_width}x{original_height}, {fps} FPS, {total_frames} frames")

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"Read {len(frames)} frames")

    # Preprocess frames
    processed_frames = preprocess_frames(frames, args.height, args.width)

    # Create frame sequences
    sequences = create_frame_sequences(processed_frames, args.sequence_length)

    print(f"Created {len(sequences)} sequences")

    # Run inference
    predictions = []

    with torch.no_grad():
        for i in range(0, len(sequences), args.batch_size):
            batch = sequences[i:i + args.batch_size]
            batch_tensor = torch.FloatTensor(batch).to(args.device)

            outputs = model(batch_tensor)
            batch_predictions = outputs.cpu().numpy()
            predictions.extend(batch_predictions)

    print(f"Generated {len(predictions)} predictions")

    # Postprocess predictions
    results = postprocess_predictions(
        predictions,
        original_height,
        original_width,
        args.height,
        args.width,
        args.threshold
    )

    # Save predictions to CSV
    df_data = []
    for i, result in enumerate(results):
        df_data.append({
            'Frame': i + args.sequence_length - 1,  # Adjust for sequence offset
            'Visibility': result['visibility'],
            'X': result['x'],
            'Y': result['y'],
            'Confidence': result['confidence']
        })

    # Add padding for the first sequence_length-1 frames
    for i in range(args.sequence_length - 1):
        df_data.insert(0, {
            'Frame': i,
            'Visibility': 0,
            'X': -1,
            'Y': -1,
            'Confidence': 0.0
        })

    df = pd.DataFrame(df_data)
    df.to_csv(args.csv_path, index=False)
    print(f"Predictions saved to: {args.csv_path}")

    # Create output video with predictions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (original_width, original_height))

    for i, frame in enumerate(frames):
        if i < len(df):
            row = df.iloc[i]
            if row['Visibility'] == 1:
                # Draw ball as red circle
                cv2.circle(frame, (int(row['X']), int(row['Y'])), 5, (0, 0, 255), -1)

                # Add confidence text
                confidence_text = f"Conf: {row['Confidence']:.3f}"
                cv2.putText(frame, confidence_text,
                           (int(row['X']) + 10, int(row['Y']) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out.write(frame)

    out.release()
    print(f"Output video saved to: {args.output_path}")

    # Print summary statistics
    visible_frames = df[df['Visibility'] == 1]
    print(f"\nSummary:")
    print(f"Total frames: {len(df)}")
    print(f"Frames with ball detected: {len(visible_frames)}")
    print(f"Detection rate: {len(visible_frames) / len(df) * 100:.1f}%")
    print(f"Average confidence: {visible_frames['Confidence'].mean():.3f}")

if __name__ == "__main__":
    main()