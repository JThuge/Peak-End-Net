"""
Frame Aesthetic Perceiver (Φ_IAA).

The per-frame aesthetic scorer is a frozen image aesthetic model pretrained on
AVA with a CLIP ViT-L/14 backbone. Only the MLP head (768 -> 512 -> 256 -> 10)
is loaded here and applied to the frozen CLIP frame features to produce a
10-class aesthetic distribution and an expected score per frame.

The Key Moment Discovery module then turns the per-frame score curve into a
single unified attention weight over frames (peak / valley / end aware), which
is consumed by the Peak-End Aggregation module.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class AvaAestheticHead(nn.Module):
    """
    Frozen AVA aesthetic head.

    Loads the MLP head (768 -> 512 -> 256 -> 10) from an AVA-pretrained
    checkpoint and scores each frame feature directly.

        clip_model (ViT-L/14) -> head -> softmax -> 10-class distribution

    Only the ``head.*`` weights are loaded (the CLIP visual backbone is shared
    with the frozen encoder in PeakAesNetV4, which uses the same ViT-L/14).

    Input:  frame_features [B, T, 768]
    Output: distributions  [B, T, 10] (softmax probabilities)
            frame_scores   [B, T]     (expected score, normalized to [0, 1])
    """

    def __init__(self, checkpoint_path=None, feature_dim=768):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )
        self.softmax = nn.Softmax(dim=-1)

        # When checkpoint_path is None, skip loading here: the head weights are
        # expected to be provided later via load_state_dict (e.g. from a merged
        # self-contained Peak-End Net checkpoint).
        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"AVA checkpoint not found: {checkpoint_path}"
                )

            # weights_only=False: the AVA checkpoint contains numpy objects.
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt

            # Extract the head.* weights.
            head_state_dict = {
                key[len("head."):]: value
                for key, value in state_dict.items()
                if key.startswith("head.")
            }
            if not head_state_dict:
                raise RuntimeError(
                    f"No 'head.*' keys found in AVA checkpoint: {checkpoint_path}. "
                    f"Available keys: {list(state_dict.keys())[:10]}"
                )
            self.head.load_state_dict(head_state_dict, strict=True)

        # Freeze the head.
        for param in self.head.parameters():
            param.requires_grad = False

    def forward(self, frame_features):
        """
        Args:
            frame_features: [B, T, 768] frozen CLIP frame features

        Returns:
            distributions: [B, T, 10] per-frame aesthetic distribution
            frame_scores:  [B, T] per-frame expected score, normalized to [0, 1]
        """
        batch_size, num_frames, feat_dim = frame_features.shape

        flat_features = frame_features.reshape(batch_size * num_frames, feat_dim)
        logits = self.head(flat_features)
        probs = self.softmax(logits)  # [B*T, 10]

        distributions = probs.reshape(batch_size, num_frames, 10)

        # Expected score: Σ(p_i * i), i ∈ [1..10], normalized to [0, 1].
        score_weights = torch.arange(
            1, 11, dtype=frame_features.dtype, device=frame_features.device
        )
        frame_scores = (distributions * score_weights).sum(dim=-1) / 10.0

        return distributions, frame_scores


class KeyMomentDiscovery(nn.Module):
    """
    Key Moment Discovery: from the per-frame aesthetic score curve, derive a
    single unified attention weight over frames that is peak / valley / end
    aware.

        w_t = softmax( σ(base) + α·peak_t + β·valley_t + γ·end_t )
    """

    def __init__(self, end_window=3):
        super().__init__()
        self.end_window = end_window

        # Learnable key-moment coefficients.
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.8))
        self.base_weight = nn.Parameter(torch.tensor(0.3))

    def _compute_peak_scores(self, frame_scores, video_mask):
        """Peak degree of each frame (local maximum on the score curve)."""
        left_diff = frame_scores[:, 1:] - frame_scores[:, :-1]
        left_diff = F.pad(left_diff, (1, 0), value=0.0)

        right_diff = frame_scores[:, :-1] - frame_scores[:, 1:]
        right_diff = F.pad(right_diff, (0, 1), value=0.0)

        peak_scores = F.relu(left_diff) * F.relu(right_diff)
        peak_scores = peak_scores * frame_scores

        if video_mask is not None:
            peak_scores = peak_scores * video_mask.float()

        return peak_scores

    def _compute_valley_scores(self, frame_scores, video_mask):
        """Valley degree of each frame (local minimum on the score curve)."""
        left_diff = frame_scores[:, :-1] - frame_scores[:, 1:]
        left_diff = F.pad(left_diff, (1, 0), value=0.0)

        right_diff = frame_scores[:, 1:] - frame_scores[:, :-1]
        right_diff = F.pad(right_diff, (0, 1), value=0.0)

        valley_scores = F.relu(left_diff) * F.relu(right_diff)
        valley_scores = valley_scores * (1.0 - frame_scores)

        if video_mask is not None:
            valley_scores = valley_scores * video_mask.float()

        return valley_scores

    def _compute_end_scores(self, frame_scores, video_mask):
        """End degree of each frame (exponential emphasis toward the end)."""
        batch_size, seq_len = frame_scores.shape
        device = frame_scores.device

        if video_mask is not None:
            valid_lengths = video_mask.float().sum(dim=1, keepdim=True)
        else:
            valid_lengths = torch.full(
                (batch_size, 1), seq_len, device=device, dtype=torch.float
            )

        positions = torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(0)
        distance_to_end = (valid_lengths - 1 - positions).clamp(min=0) / valid_lengths.clamp(min=1)

        decay_rate = seq_len / max(self.end_window, 1)
        end_scores = torch.exp(-decay_rate * distance_to_end)
        end_scores = end_scores * frame_scores

        if video_mask is not None:
            end_scores = end_scores * video_mask.float()

        return end_scores

    def forward(self, frame_scores, video_mask=None):
        """
        Args:
            frame_scores: [B, T] per-frame aesthetic score (in [0, 1])
            video_mask:   [B, T] video mask

        Returns:
            attention_weights: [B, T] unified per-frame attention weight
            moment_info: dict with peak / valley / end components
        """
        peak_scores = self._compute_peak_scores(frame_scores, video_mask)
        valley_scores = self._compute_valley_scores(frame_scores, video_mask)
        end_scores = self._compute_end_scores(frame_scores, video_mask)

        base = torch.sigmoid(self.base_weight)
        combined_importance = (
            base
            + torch.abs(self.alpha) * peak_scores
            + torch.abs(self.beta) * valley_scores
            + torch.abs(self.gamma) * end_scores
        )

        if video_mask is not None:
            combined_importance = combined_importance.masked_fill(
                video_mask == 0, float("-inf")
            )

        attention_weights = F.softmax(combined_importance, dim=-1)

        moment_info = {
            "peak_scores": peak_scores,
            "valley_scores": valley_scores,
            "end_scores": end_scores,
            "raw_importance": combined_importance,
        }

        return attention_weights, moment_info
