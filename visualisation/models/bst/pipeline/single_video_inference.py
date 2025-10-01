#!/usr/bin/env python3
"""
ABOUTME: Single video BST inference with visualization for badminton stroke analysis
ABOUTME: Runs trained BST model on specific video and shows detailed stroke-by-stroke results
"""

import sys
from pathlib import Path
# Add the parent directory to Python path so we can import from models
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import cv2
from collections import defaultdict
import argparse

from models.bst import BST_8
from models.dataset import Dataset_npy, get_bone_pairs, create_bones
from torch.utils.data import DataLoader

class SingleVideoAnalyzer:
    def __init__(self, weight_path, data_root, device='cpu'):
        self.weight_path = Path(weight_path)
        self.data_root = Path(data_root)
        self.device = device
        
        # Load trained model
        print(f"🔄 Loading trained BST model from {weight_path}")
        # BST_8 parameters: in_dim, seq_len, n_class, n_people
        # in_dim = (n_joints + n_bones * extra) * in_channels = (17 + 19 * 1) * 2 = 72
        self.model = BST_8(in_dim=72, seq_len=30, n_class=35, depth_tem=2, depth_inter=1)
        self.model.load_state_dict(torch.load(str(weight_path), map_location=device, weights_only=True))
        self.model.to(device)
        self.model.eval()
        print(f"✅ Model loaded successfully")
        
        # Class names mapping
        self.class_names = [
            'Top_放小球', 'Top_擋小球', 'Top_殺球', 'Top_點扣', 'Top_挑球', 'Top_防守回挑', 
            'Top_長球', 'Top_平球', 'Top_後場抽平球', 'Top_切球', 'Top_過渡切球', 'Top_推球',
            'Top_撲球', 'Top_防守回抽', 'Top_勾球', 'Top_發短球', 'Top_發長球',
            'Bottom_放小球', 'Bottom_擋小球', 'Bottom_殺球', 'Bottom_點扣', 'Bottom_挑球', 
            'Bottom_防守回挑', 'Bottom_長球', 'Bottom_平球', 'Bottom_後場抽平球', 'Bottom_切球',
            'Bottom_過渡切球', 'Bottom_推球', 'Bottom_撲球', 'Bottom_防守回抽', 'Bottom_勾球',
            'Bottom_發短球', 'Bottom_發長球', '未知球種'
        ]
    
    def find_available_videos(self):
        """Find all available videos in the test set"""
        video_info = defaultdict(lambda: {'total_strokes': 0, 'strokes': []})
        
        print("🔍 Scanning test data for available videos...")
        
        for stroke_dir in self.data_root.iterdir():
            if stroke_dir.is_dir() and stroke_dir.name != 'test_specific':
                stroke_type = stroke_dir.name
                
                for file_path in stroke_dir.glob("*_joints.npy"):
                    # Parse filename: {video_id}_{set}_{rally}_{shot}_{player}_{stroke_type}_joints.npy
                    parts = file_path.stem.split('_')
                    if len(parts) >= 6:  # Need at least video, set, rally, shot, player, stroke_type
                        video_id = parts[0]
                        set_id = parts[1] 
                        rally_id = parts[2]
                        shot_id = parts[3]
                        player = parts[4]
                        
                        video_key = f"Video_{video_id}_Set_{set_id}"
                        video_info[video_key]['total_strokes'] += 1
                        video_info[video_key]['strokes'].append({
                            'rally': rally_id,
                            'shot': shot_id,
                            'player': player,
                            'stroke_type': stroke_type,
                            'file_prefix': '_'.join(parts[:-1])  # Remove _joints suffix only
                        })
        
        # Sort by total strokes for better display
        sorted_videos = sorted(video_info.items(), key=lambda x: x[1]['total_strokes'], reverse=True)
        
        print(f"📺 Found {len(sorted_videos)} videos with stroke data:")
        print("-" * 60)
        for i, (video_key, info) in enumerate(sorted_videos[:10]):  # Show top 10
            print(f"{i+1:2d}. {video_key}: {info['total_strokes']} strokes")
        
        if len(sorted_videos) > 10:
            print(f"    ... and {len(sorted_videos) - 10} more videos")
        
        return dict(sorted_videos)
    
    def analyze_video(self, video_key):
        """Analyze all strokes from a specific video"""
        video_info = self.find_available_videos()
        
        if video_key not in video_info:
            print(f"❌ Video {video_key} not found!")
            return None
        
        strokes_data = video_info[video_key]['strokes']
        print(f"\n🎬 Analyzing {video_key} with {len(strokes_data)} strokes")
        print("=" * 60)
        
        results = []
        
        for i, stroke in enumerate(strokes_data):
            print(f"🏸 Analyzing stroke {i+1}/{len(strokes_data)}: Rally {stroke['rally']}, Shot {stroke['shot']}")
            
            # Load stroke data
            stroke_result = self.analyze_single_stroke(
                stroke['file_prefix'], 
                stroke['stroke_type']
            )
            
            if stroke_result:
                stroke_result.update({
                    'rally': stroke['rally'],
                    'shot': stroke['shot'],
                    'ground_truth': stroke['stroke_type']
                })
                results.append(stroke_result)
        
        # Generate video report
        self.generate_video_report(video_key, results)
        return results
    
    def analyze_single_stroke(self, file_prefix, ground_truth_stroke):
        """Analyze a single stroke given its file prefix"""
        try:
            # Construct file paths - the file_prefix already includes the full name except the suffix
            base_path = None
            
            # Find the stroke in the correct directory
            for stroke_dir in self.data_root.iterdir():
                if stroke_dir.is_dir() and stroke_dir.name == ground_truth_stroke:
                    joints_file = stroke_dir / f"{file_prefix}_joints.npy"
                    pos_file = stroke_dir / f"{file_prefix}_pos.npy" 
                    shuttle_file = stroke_dir / f"{file_prefix}_shuttle.npy"
                    
                    if all(f.exists() for f in [joints_file, pos_file, shuttle_file]):
                        base_path = stroke_dir / file_prefix
                        break
            
            if not base_path:
                print(f"⚠️  Could not find files for {file_prefix}")
                return None
            
            # Load the three data components using the same format as training
            joints_data = np.load(f"{base_path}_joints.npy").astype(np.float32)      # (seq_len, 2, 17, 2)
            pos_data = np.load(f"{base_path}_pos.npy").astype(np.float32)            # (seq_len, 2, 2)  
            shuttle_data = np.load(f"{base_path}_shuttle.npy").astype(np.float32)    # (seq_len, 3)
            
            # Process pose data using JnB_bone style (same as training)
            bone_pairs = get_bone_pairs()
            bones_data = create_bones(joints_data, bone_pairs)  # Create bone features
            human_pose = np.concatenate((joints_data, bones_data), axis=-2)  # Combine joints + bones
            
            # Create synthetic video length (all frames are valid)
            video_len = torch.tensor([30], dtype=torch.long).to(self.device)
            
            # Convert to tensors and add batch dimension
            human_pose_tensor = torch.FloatTensor(human_pose).unsqueeze(0).to(self.device)  # (1, seq_len, 2, 17+19, 2)
            pos_tensor = torch.FloatTensor(pos_data).unsqueeze(0).to(self.device)           # (1, seq_len, 2, 2)
            shuttle_tensor = torch.FloatTensor(shuttle_data).unsqueeze(0).to(self.device)    # (1, seq_len, 3)
            
            # Reshape for model input - flatten the last dimensions as done in training
            human_pose_flat = human_pose_tensor.view(*human_pose_tensor.shape[:-2], -1)  # (1, seq_len, 72)
            
            # Run inference
            with torch.no_grad():
                output = self.model(human_pose_flat, shuttle_tensor, pos_tensor, video_len)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                
                # Get top 3 predictions
                top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
                top3_predictions = [
                    {
                        'class': self.class_names[idx.item()], 
                        'confidence': prob.item()
                    } 
                    for idx, prob in zip(top3_indices[0], top3_probs[0])
                ]
            
            # Determine if prediction is correct
            gt_class_idx = self.class_names.index(ground_truth_stroke) if ground_truth_stroke in self.class_names else -1
            is_correct = predicted_class == gt_class_idx
            
            return {
                'file_prefix': file_prefix,
                'predicted_class': self.class_names[predicted_class],
                'predicted_class_idx': predicted_class,
                'confidence': confidence,
                'is_correct': is_correct,
                'top3_predictions': top3_predictions,
                'joints_data': joints_data,
                'pos_data': pos_data, 
                'shuttle_data': shuttle_data
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {file_prefix}: {str(e)}")
            return None
    
    def generate_video_report(self, video_key, results):
        """Generate comprehensive report for video analysis"""
        if not results:
            print("❌ No results to analyze")
            return
        
        print(f"\n📊 ANALYSIS REPORT FOR {video_key}")
        print("=" * 80)
        
        # Overall statistics
        total_strokes = len(results)
        correct_predictions = sum(1 for r in results if r['is_correct'])
        accuracy = correct_predictions / total_strokes
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"📈 Overall Performance:")
        print(f"   • Total Strokes: {total_strokes}")
        print(f"   • Correct Predictions: {correct_predictions}/{total_strokes}")
        print(f"   • Accuracy: {accuracy:.1%}")
        print(f"   • Average Confidence: {avg_confidence:.3f}")
        
        # Stroke-by-stroke breakdown
        print(f"\n🏸 Stroke-by-Stroke Analysis:")
        print("-" * 80)
        print(f"{'#':<3} {'Rally':<6} {'Shot':<5} {'Ground Truth':<20} {'Prediction':<20} {'Conf':<6} {'✓'}")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            status = "✅" if result['is_correct'] else "❌"
            print(f"{i:<3} {result['rally']:<6} {result['shot']:<5} "
                  f"{result['ground_truth']:<20} {result['predicted_class']:<20} "
                  f"{result['confidence']:.3f} {status}")
        
        # Confusion analysis
        print(f"\n🎯 Prediction Analysis:")
        stroke_types = {}
        for result in results:
            gt = result['ground_truth']
            pred = result['predicted_class']
            
            if gt not in stroke_types:
                stroke_types[gt] = {'total': 0, 'correct': 0, 'predictions': []}
            
            stroke_types[gt]['total'] += 1
            stroke_types[gt]['predictions'].append(pred)
            if result['is_correct']:
                stroke_types[gt]['correct'] += 1
        
        for stroke_type, stats in stroke_types.items():
            accuracy = stats['correct'] / stats['total']
            print(f"   • {stroke_type}: {stats['correct']}/{stats['total']} ({accuracy:.1%})")
        
        # Save detailed results
        self.save_video_results(video_key, results)
        
        # Generate visualizations
        self.create_video_visualizations(video_key, results)
        
        print(f"\n💾 Results saved to video_analysis_{video_key.lower().replace(' ', '_')}.json")
        print(f"📊 Visualizations saved as PNG files")
    
    def save_video_results(self, video_key, results):
        """Save detailed results to JSON file"""
        output_data = {
            'video': video_key,
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_strokes': len(results),
            'accuracy': sum(1 for r in results if r['is_correct']) / len(results),
            'strokes': []
        }
        
        for result in results:
            stroke_data = {
                'rally': result['rally'],
                'shot': result['shot'],
                'ground_truth': result['ground_truth'],
                'prediction': result['predicted_class'], 
                'confidence': float(result['confidence']),
                'is_correct': result['is_correct'],
                'top3_predictions': [
                    {'class': p['class'], 'confidence': float(p['confidence'])} 
                    for p in result['top3_predictions']
                ]
            }
            output_data['strokes'].append(stroke_data)
        
        filename = f"video_analysis_{video_key.lower().replace(' ', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def create_video_visualizations(self, video_key, results):
        """Create visualization plots for the video analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'BST Analysis: {video_key}', fontsize=16, fontweight='bold')
        
        # 1. Confidence distribution
        confidences = [r['confidence'] for r in results]
        correct_conf = [r['confidence'] for r in results if r['is_correct']]
        incorrect_conf = [r['confidence'] for r in results if not r['is_correct']]
        
        axes[0,0].hist([correct_conf, incorrect_conf], bins=20, alpha=0.7, 
                      label=['Correct', 'Incorrect'], color=['green', 'red'])
        axes[0,0].set_xlabel('Confidence Score')
        axes[0,0].set_ylabel('Count')
        axes[0,0].set_title('Prediction Confidence Distribution')
        axes[0,0].legend()
        
        # 2. Stroke timeline
        stroke_indices = list(range(len(results)))
        colors = ['green' if r['is_correct'] else 'red' for r in results]
        axes[0,1].scatter(stroke_indices, confidences, c=colors, alpha=0.7)
        axes[0,1].set_xlabel('Stroke Number')
        axes[0,1].set_ylabel('Confidence')
        axes[0,1].set_title('Prediction Timeline')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Stroke type performance
        stroke_performance = {}
        for result in results:
            gt = result['ground_truth']
            if gt not in stroke_performance:
                stroke_performance[gt] = {'correct': 0, 'total': 0}
            stroke_performance[gt]['total'] += 1
            if result['is_correct']:
                stroke_performance[gt]['correct'] += 1
        
        stroke_names = list(stroke_performance.keys())
        accuracies = [stroke_performance[name]['correct'] / stroke_performance[name]['total'] 
                     for name in stroke_names]
        
        if len(stroke_names) <= 10:  # Only plot if reasonable number
            axes[1,0].bar(range(len(stroke_names)), accuracies)
            axes[1,0].set_xlabel('Stroke Type')
            axes[1,0].set_ylabel('Accuracy')
            axes[1,0].set_title('Per Stroke Type Accuracy')
            axes[1,0].set_xticks(range(len(stroke_names)))
            axes[1,0].set_xticklabels(stroke_names, rotation=45, ha='right')
        else:
            axes[1,0].text(0.5, 0.5, f'Too many stroke types\nto display ({len(stroke_names)})', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
        
        # 4. Top-3 accuracy
        top1_acc = sum(1 for r in results if r['is_correct']) / len(results)
        
        # Calculate top-3 accuracy
        top3_correct = 0
        for result in results:
            gt = result['ground_truth']
            top3_classes = [p['class'] for p in result['top3_predictions']]
            if gt in top3_classes:
                top3_correct += 1
        top3_acc = top3_correct / len(results)
        
        acc_data = [top1_acc, top3_acc]
        axes[1,1].bar(['Top-1', 'Top-3'], acc_data, color=['skyblue', 'lightcoral'])
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_title('Top-K Accuracy')
        axes[1,1].set_ylim(0, 1)
        
        # Add accuracy text on bars
        for i, acc in enumerate(acc_data):
            axes[1,1].text(i, acc + 0.02, f'{acc:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"video_analysis_{video_key.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze BST performance on single badminton video')
    parser.add_argument('--video', type=str, help='Video key to analyze (e.g., "Video_13_Set_1")')
    parser.add_argument('--list-videos', action='store_true', help='List available videos')
    parser.add_argument('--weight', type=str, 
                       default='weight/bst_8_JnB_bone_between_2_hits_with_max_limits_seq_100_11.pt',
                       help='Path to trained model weights')
    parser.add_argument('--data', type=str, default='dataset_npy_bst_format/test',
                       help='Path to test data directory')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyzer = SingleVideoAnalyzer(args.weight, args.data, device)
    
    if args.list_videos:
        # Just show available videos
        analyzer.find_available_videos()
        return
    
    if args.video:
        # Analyze specific video
        results = analyzer.analyze_video(args.video)
    else:
        # Interactive mode - let user choose video
        print("🎬 Interactive Video Selection Mode")
        videos = analyzer.find_available_videos()
        
        print("\nSelect a video to analyze:")
        video_list = list(videos.keys())
        for i, video_key in enumerate(video_list[:20]):  # Show first 20
            print(f"{i+1:2d}. {video_key} ({videos[video_key]['total_strokes']} strokes)")
        
        try:
            choice = int(input(f"\nEnter choice (1-{min(len(video_list), 20)}): ")) - 1
            if 0 <= choice < len(video_list):
                selected_video = video_list[choice]
                print(f"\n🎯 Selected: {selected_video}")
                results = analyzer.analyze_video(selected_video)
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")


if __name__ == '__main__':
    main()