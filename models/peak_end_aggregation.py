"""
Peak-End Aggregation (temporal aggregation).

Given the per-frame attention weights produced by Key Moment Discovery, the
video-level feature is the weighted average of frame features followed by a
projection:

    f_video = Σ_t w_t · f_t   ->   proj   ->   LayerNorm
"""

import torch
import torch.nn as nn


class PeakEndAggregation(nn.Module):
    """
    Temporal aggregation via a single unified attention weight.

    forward receives ``attention_weights`` (from KeyMomentDiscovery) and does a
    weighted pooling over frame features, then projects to ``output_dim``.
    """

    def __init__(self, embed_dim=768, output_dim=768):
        super().__init__()
        self.output_dim = output_dim

        # Projection: map the weighted-pooled feature to output_dim.
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, frame_features, attention_weights):
        """
        Args:
            frame_features:    [B, T, embed_dim] frame features
            attention_weights: [B, T] per-frame attention weight

        Returns:
            video_feature: [B, output_dim] aggregated video feature
        """
        weighted_features = torch.bmm(
            attention_weights.unsqueeze(1),  # [B, 1, T]
            frame_features,                  # [B, T, embed_dim]
        ).squeeze(1)                         # [B, embed_dim]

        video_feature = self.proj(weighted_features)
        video_feature = self.output_norm(video_feature)

        return video_feature
