"""
Peak-End Net inference script.

Loads the self-contained checkpoint (e.g. Peak-End-Net.pth from
https://huggingface.co/GD-ML/Peak-End-Net) and predicts the 11 aesthetic
attribute scores for a single video.

The checkpoint bundles the full model weights (CLIP ViT-L/14 encoder, AVA
aesthetic head, Peak-End modules and the gated fusion module), so no external
AVA / Stage-1 checkpoints are required.

Usage:
    python inference.py \
        --checkpoint ./checkpoints/Peak-End-Net.pth \
        --video /path/to/video.mp4
"""

import argparse

import numpy as np
import torch

from models.gated_fusion import GatedFusionPeakAesNetV4
from modules.rawvideo_util import RawVideoExtractor

# 11 aesthetic attribute names (index 0 = overall).
SCORE_NAMES = [
    "overall", "composition", "shotsize", "lighting",
    "visualtone", "color", "depthoffield",
    "expression", "movement", "costume", "makeup",
]


def load_model(checkpoint_path, device, pretrained_clip_name="ViT-L/14"):
    """Build the model skeleton and load the merged self-contained weights."""
    model = GatedFusionPeakAesNetV4(
        stage1_checkpoint_path=None,  # weights come from the merged checkpoint
        ava_checkpoint_path=None,     # weights come from the merged checkpoint
        pretrained_clip_name=pretrained_clip_name,
        max_frames=12,
        frame_context_window=3,
        end_window=3,
        rhythm_dim=64,
        end_ratio=0.25,
        scoring_hidden_dim=256,
        fusion_hidden_dim=128,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

    model.to(device).eval()
    return model


def preprocess_video(video_path, max_frames=12, image_resolution=224, slice_framepos=2):
    """Decode a video into the model input tensor (same pipeline as training).

    Returns:
        video:      [1, 1, T, 1, 3, H, W] float tensor
        video_mask: [1, T] long tensor
    """
    extractor = RawVideoExtractor(framerate=1, size=image_resolution)

    video = np.zeros(
        (1, max_frames, 1, 3, image_resolution, image_resolution),
        dtype=np.float32,
    )
    video_mask = np.zeros((1, max_frames), dtype=np.int64)
    max_len = 0

    raw = extractor.get_video_data(video_path)["video"]
    if len(raw.shape) > 3:
        raw_slice = extractor.process_raw_data(raw)
        if max_frames < raw_slice.shape[0]:
            if slice_framepos == 0:
                sel = raw_slice[:max_frames, ...]
            elif slice_framepos == 1:
                sel = raw_slice[-max_frames:, ...]
            else:
                idx = np.linspace(0, raw_slice.shape[0] - 1, num=max_frames, dtype=int)
                sel = raw_slice[idx, ...]
        else:
            sel = raw_slice
        sel = extractor.process_frame_order(sel, frame_order=0)
        max_len = sel.shape[0]
        if max_len >= 1:
            video[0][:max_len, ...] = sel

    video_mask[0][:max_len] = 1

    # [1, T, 1, 3, H, W] -> [1, 1, T, 1, 3, H, W] (batch dim)
    video_tensor = torch.tensor(video).unsqueeze(0)
    mask_tensor = torch.tensor(video_mask)
    return video_tensor, mask_tensor


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Peak-End Net inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Peak-End-Net.pth (self-contained checkpoint)")
    parser.add_argument("--video", type=str, required=True, help="Path to a video file")
    parser.add_argument("--pretrained_clip_name", type=str, default="ViT-L/14")
    parser.add_argument("--max_frames", type=int, default=12)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, device, args.pretrained_clip_name)

    print(f"Preprocessing video: {args.video}")
    video, video_mask = preprocess_video(args.video, max_frames=args.max_frames)
    video = video.to(device)
    video_mask = video_mask.to(device)

    final_scores, aux = model(video, video_mask)
    scores = final_scores[0].cpu().numpy()

    print("\n================ Aesthetic Scores ================")
    for name, value in zip(SCORE_NAMES, scores):
        print(f"  {name:<12}: {value:.4f}")
    print("--------------------------------------------------")
    print(f"  gate     : {aux['gate'][0].item():.4f}")
    print(f"  S_model  : {aux['S_model'][0].item():.4f}")
    print(f"  S_static : {aux['S_static'][0].item():.4f}")
    print("==================================================")


if __name__ == "__main__":
    main()
