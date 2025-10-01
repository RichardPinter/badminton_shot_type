# ABOUTME: Utility functions for TrackNet training and evaluation
# ABOUTME: Includes data preprocessing, model training, evaluation metrics, and visualization tools

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms
import glob

# Data preprocessing utilities
def listdir_fullpath(d):
    """Get full paths of all files in directory"""
    return [os.path.join(d, f) for f in os.listdir(d)]

def get_list(directory):
    """Get sorted list of files in directory"""
    files = sorted(glob.glob(os.path.join(directory, '*')))
    return files

def get_video_frames(video_path):
    """Extract frames from video file"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def resize_frame(frame, height=288, width=512):
    """Resize frame to specified dimensions"""
    return cv2.resize(frame, (width, height))

def normalize_frame(frame):
    """Normalize frame pixel values to [0, 1]"""
    return frame.astype(np.float32) / 255.0

# Model utilities
def get_model_summary(model, input_size=(9, 288, 512)):
    """Print model summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input size: {input_size}")

    # Test forward pass
    x = torch.randn(1, *input_size)
    with torch.no_grad():
        output = model(x)
    print(f"Output size: {output.shape}")

    return model

# Custom loss functions
class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, pos_weight=1.0):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, output, target):
        loss = F.binary_cross_entropy_with_logits(
            output, target,
            pos_weight=torch.tensor(self.pos_weight)
        )
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return loss.mean()

# Training utilities
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0:
            print(f'Epoch {epoch}, Batch {i}/{num_batches}, Loss: {loss.item():.4f}')

    avg_loss = running_loss / num_batches
    return avg_loss

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

            # Convert to predictions
            predictions = torch.sigmoid(outputs) > 0.5
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='binary')
    recall = recall_score(all_targets, all_predictions, average='binary')
    f1 = f1_score(all_targets, all_predictions, average='binary')

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Prediction utilities
def predict_ball_location(model, frame_sequence, device, threshold=0.5):
    """Predict ball location in frame sequence"""
    model.eval()

    with torch.no_grad():
        # Ensure input is tensor
        if isinstance(frame_sequence, np.ndarray):
            frame_sequence = torch.FloatTensor(frame_sequence)

        if len(frame_sequence.shape) == 3:
            frame_sequence = frame_sequence.unsqueeze(0)  # Add batch dimension

        frame_sequence = frame_sequence.to(device)

        output = model(frame_sequence)
        heatmap = torch.sigmoid(output).cpu().numpy()[0, 0]  # Get first channel of first batch

        # Find maximum location
        max_val = np.max(heatmap)

        if max_val > threshold:
            max_loc = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            return {
                'visibility': 1,
                'x': int(max_loc[1]),
                'y': int(max_loc[0]),
                'confidence': float(max_val)
            }
        else:
            return {
                'visibility': 0,
                'x': -1,
                'y': -1,
                'confidence': float(max_val)
            }

def draw_ball_on_frame(frame, prediction, original_size=None, model_size=(288, 512)):
    """Draw predicted ball location on frame"""
    if prediction['visibility'] == 0:
        return frame

    # Scale coordinates if needed
    if original_size is not None:
        orig_h, orig_w = original_size
        model_h, model_w = model_size

        x = int(prediction['x'] * orig_w / model_w)
        y = int(prediction['y'] * orig_h / model_h)
    else:
        x = prediction['x']
        y = prediction['y']

    # Draw circle
    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # Add confidence text
    confidence_text = f"{prediction['confidence']:.2f}"
    cv2.putText(frame, confidence_text, (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

# Visualization utilities
def plot_training_history(train_losses, val_losses=None, val_metrics=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    axes[0].plot(train_losses, label='Train Loss')
    if val_losses is not None:
        axes[0].plot(val_losses, label='Validation Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot metrics
    if val_metrics is not None:
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics_to_plot:
            if metric in val_metrics:
                metric_values = [m[metric] for m in val_metrics]
                axes[1].plot(metric_values, label=metric.capitalize())

        axes[1].set_title('Validation Metrics')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    return fig

def visualize_prediction(frame, prediction, ground_truth=None):
    """Visualize prediction on frame"""
    vis_frame = frame.copy()

    # Draw prediction in red
    if prediction['visibility'] == 1:
        cv2.circle(vis_frame, (prediction['x'], prediction['y']), 5, (0, 0, 255), -1)
        cv2.putText(vis_frame, f"Pred: {prediction['confidence']:.2f}",
                   (prediction['x'] + 10, prediction['y'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw ground truth in green
    if ground_truth is not None and ground_truth['visibility'] == 1:
        cv2.circle(vis_frame, (ground_truth['x'], ground_truth['y']), 5, (0, 255, 0), 2)
        cv2.putText(vis_frame, "GT",
                   (ground_truth['x'] + 10, ground_truth['y'] + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return vis_frame

# Data generation utilities
def generate_heatmap(center, height, width, sigma=5):
    """Generate Gaussian heatmap for ball location"""
    x, y = center
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Create meshgrid
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate Gaussian
    gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    return gaussian

def create_frame_sequence(frames, sequence_length=3):
    """Create frame sequence for model input"""
    if len(frames) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} frames, got {len(frames)}")

    # Take the last sequence_length frames
    sequence = frames[-sequence_length:]

    # Stack frames channel-wise
    stacked = np.concatenate(sequence, axis=2)

    # Transpose to CHW format for PyTorch
    stacked = np.transpose(stacked, (2, 0, 1))

    return stacked

# File I/O utilities
def save_predictions_csv(predictions, output_path):
    """Save predictions to CSV file"""
    data = []
    for i, pred in enumerate(predictions):
        data.append({
            'Frame': i,
            'Visibility': pred['visibility'],
            'X': pred['x'],
            'Y': pred['y'],
            'Confidence': pred['confidence']
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return df

def load_predictions_csv(csv_path):
    """Load predictions from CSV file"""
    df = pd.read_csv(csv_path)
    predictions = []

    for _, row in df.iterrows():
        predictions.append({
            'visibility': int(row['Visibility']),
            'x': int(row['X']) if row['X'] != -1 else -1,
            'y': int(row['Y']) if row['Y'] != -1 else -1,
            'confidence': float(row['Confidence'])
        })

    return predictions

# Performance metrics
def calculate_distance_error(pred, gt):
    """Calculate distance between predicted and ground truth positions"""
    if pred['visibility'] == 0 or gt['visibility'] == 0:
        return None

    dx = pred['x'] - gt['x']
    dy = pred['y'] - gt['y']
    return np.sqrt(dx ** 2 + dy ** 2)

def evaluate_tracking_performance(predictions, ground_truths, distance_threshold=10):
    """Evaluate tracking performance with various metrics"""
    total_frames = len(predictions)
    correct_detections = 0
    false_positives = 0
    false_negatives = 0
    distance_errors = []

    for pred, gt in zip(predictions, ground_truths):
        if gt['visibility'] == 1:  # Ground truth has ball
            if pred['visibility'] == 1:  # Prediction has ball
                distance = calculate_distance_error(pred, gt)
                if distance is not None:
                    distance_errors.append(distance)
                    if distance <= distance_threshold:
                        correct_detections += 1
                    else:
                        false_positives += 1
            else:  # Prediction missed ball
                false_negatives += 1
        else:  # Ground truth has no ball
            if pred['visibility'] == 1:  # False positive
                false_positives += 1

    precision = correct_detections / (correct_detections + false_positives) if (correct_detections + false_positives) > 0 else 0
    recall = correct_detections / (correct_detections + false_negatives) if (correct_detections + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_distance_error': np.mean(distance_errors) if distance_errors else 0,
        'std_distance_error': np.std(distance_errors) if distance_errors else 0,
        'correct_detections': correct_detections,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_frames': total_frames
    }

    return results