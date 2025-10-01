#!/usr/bin/env python3
import sys
from pathlib import Path
# Add the parent directory to Python path so we can import from models
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
import torch.nn.functional as F

# repo-local imports
from models.bst import BST_8
from models.dataset import get_stroke_types, get_bone_pairs  # bone pairs used to build JnB_bone

SEQ_LEN = 100  # Data sequence length (BST_8 internally uses SEQ_LEN+1=101 for embeddings)
N_PLAYERS = 2

# Frontier classes for bottom court (6 classes)
FRONTIER_CLASSES = ['clear', 'drive', 'drop', 'lob', 'net', 'smash']

def load_npy(path: Path) -> np.ndarray:
    return np.load(str(path))

def ensure_seq_len(x: np.ndarray) -> np.ndarray:
    """Ensure sequence has length SEQ_LEN (pad or crop as needed)"""
    t = x.shape[0]
    if t == SEQ_LEN:
        return x
    if t < SEQ_LEN:
        pad = np.repeat(x[-1:], SEQ_LEN - t, axis=0)
        return np.concatenate([x, pad], axis=0)
    start = max(0, (t - SEQ_LEN) // 2)
    return x[start:start + SEQ_LEN]

def ensure_pose_JnB_bone(joints_np: np.ndarray) -> np.ndarray:
    """
    Make sure pose is (T, P, J+B, 2). If we only have joints (J,2) per player,
    compute bones with get_bone_pairs() and concat -> (J+B,2).
    Tolerates either (T,P,J,2) or (T,P,J*2).
    """
    # If flattened, unflatten to (T,P,J,2)
    if joints_np.ndim == 3:
        T, P, flat = joints_np.shape
        assert P == N_PLAYERS, f"Expected {N_PLAYERS} players, got {P}"
        assert flat % 2 == 0, "Pose last dim not even; can't reshape to (..., 2)"
        J = flat // 2
        joints_np = joints_np.reshape(T, P, J, 2)
    elif joints_np.ndim == 4:
        T, P, J, C = joints_np.shape
        assert P == N_PLAYERS, f"Expected {N_PLAYERS} players, got {P}"
        assert C == 2, f"Expected last dim=2 (x,y), got {C}"
    else:
        raise ValueError(f"Unexpected joints shape {joints_np.shape}")

    # If we already have J+B (e.g., 36), return as-is
    J = joints_np.shape[2]
    # Typical COCO: J=17; bones ~19 → J+B=36
    # Heuristic: if J>=30 we assume it's already J+B
    if J >= 30:
        return joints_np

    # Build bones
    pairs = get_bone_pairs()  # default coco pairs used in training
    bones = []
    # bone = end - start
    for (a, b) in pairs:
        bones.append(joints_np[:, :, b, :] - joints_np[:, :, a, :])
    bones_np = np.stack(bones, axis=2)  # (T,P,B,2)

    jnb = np.concatenate([joints_np, bones_np], axis=2)  # (T,P,J+B,2)
    return jnb

@torch.no_grad()
def infer_triplet(joints_np: np.ndarray,
                  pos_np: np.ndarray,
                  shuttle_np: np.ndarray,
                  weight_path: Path,
                  device: str = "cpu"):
    joints_np = ensure_seq_len(joints_np)
    pos_np = ensure_seq_len(pos_np)
    shuttle_np = ensure_seq_len(shuttle_np)

    # Ensure pose is JnB_bone (T,P,J+B,2)
    joints_np = ensure_pose_JnB_bone(joints_np)

    # Torch tensors (batch size = 1)
    device = torch.device(device)
    human_pose = torch.tensor(joints_np, dtype=torch.float32, device=device)  # (T,P,J+B,2)
    pos = torch.tensor(pos_np, dtype=torch.float32, device=device)            # (T,P,2)
    shuttle = torch.tensor(shuttle_np, dtype=torch.float32, device=device)    # (T,2)
    video_len = torch.tensor([SEQ_LEN], dtype=torch.int32, device=device)     # (1,)

    # Flatten pose for the model: (B,T,P,(J+B)*2)
    human_pose = human_pose.unsqueeze(0)  # (1,T,P,J+B,2)
    B, T, P, JplusB, C = human_pose.shape
    human_pose = human_pose.view(B, T, P, JplusB * C)

    pos = pos.unsqueeze(0)         # (1,T,P,2)
    shuttle = shuttle.unsqueeze(0) # (1,T,2)

    # Build BST_8 — in_dim=(J+B)*2
    in_dim = JplusB * C  # expected 72 for COCO JnB_bone
    n_classes = len(FRONTIER_CLASSES)  # Use frontier classes (6 classes)
    net = BST_8(
        in_dim=in_dim,
        n_class=n_classes,
        seq_len=SEQ_LEN,
        depth_tem=2,
        depth_inter=1
    ).to(device)

    # Load weights
    ckpt = torch.load(str(weight_path), map_location=device, weights_only=False)
    state = ckpt if isinstance(ckpt, dict) and "state_dict" not in ckpt else ckpt.get("state_dict", ckpt)
    net.load_state_dict(state, strict=True)
    net.eval()

    logits = net(human_pose, shuttle, pos, video_len)  # (1, n_classes)
    probs = F.softmax(logits, dim=1).squeeze(0)        # (n_classes,)
    conf, idx = torch.topk(probs, k=3)

    classes = FRONTIER_CLASSES  # Use frontier classes
    top3 = [(classes[i], float(conf[j].item())) for j, i in enumerate(idx.tolist())]
    pred_idx = int(torch.argmax(probs).item())
    pred_class = classes[pred_idx]
    pred_conf = float(probs[pred_idx].item())
    return pred_class, pred_conf, top3

def main():
    ap = argparse.ArgumentParser(description="Run BST_8 on a single triplet of npy files.")
    ap.add_argument("base", type=str,
                    help="Base path WITHOUT suffix, e.g. /tmp/.../muted_13_1_2_5 "
                         "(script will append _joints.npy, _pos.npy, _shuttle.npy)")
    ap.add_argument("--weights", type=str,
                    default="weight/bst_8_JnB_bone_between_2_hits_with_max_limits_seq_100_11.pt",
                    help="Path to BST weights .pt")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                    help="Device to run inference on")
    args = ap.parse_args()

    base = Path(args.base)
    joints_p = base.parent / (base.name + "_joints.npy")
    pos_p = base.parent / (base.name + "_pos.npy")
    shuttle_p = base.parent / (base.name + "_shuttle.npy")

    for p in (joints_p, pos_p, shuttle_p):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    joints = load_npy(joints_p)
    pos = load_npy(pos_p)
    shuttle = load_npy(shuttle_p)

    pred_class, pred_conf, top3 = infer_triplet(
        joints_np=joints,
        pos_np=pos,
        shuttle_np=shuttle,
        weight_path=Path(args.weights),
        device=args.device
    )

    print("\n🎯 Prediction")
    print(f"Class: {pred_class}")
    print(f"Confidence: {pred_conf:.3f}")
    print("\nTop-3:")
    for i, (c, s) in enumerate(top3, 1):
        bar = "█" * int(s * 20)
        print(f"{i}. {c:>12s}  {s:6.2%}  {bar}")

if __name__ == "__main__":
    main()
