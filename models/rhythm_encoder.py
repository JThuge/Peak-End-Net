"""
消融实验：美学节奏编码器（移除曲线统计特征）

与原版 experiments_peakaes_v4/aesthetic_rhythm_encoder.py 的区别：
  - 移除了 AestheticCurveStatistics 模块（10 维统计特征）
  - 只保留 RhythmPatternEncoder（1D CNN 节奏模式特征）
  - output_dim: 74 → 64

目的：验证曲线统计特征（均值、标准差、斜率、峰谷差、结尾趋势等）
对最终性能的贡献。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RhythmPatternEncoder(nn.Module):
    """
    节奏模式编码器：用 1D CNN 对帧级美学分数序列做卷积，
    学习更复杂的时序模式特征。

    架构：
        [B, 1, T] → Conv1D layers → Global Pool → [B, rhythm_dim]

    多尺度卷积核捕捉不同时间跨度的节奏模式：
    - 小核（3）：捕捉局部波动
    - 中核（5）：捕捉中程趋势
    - 大核（7）：捕捉整体节奏
    """

    def __init__(self, max_seq_len=32, rhythm_dim=64):
        super().__init__()
        self.rhythm_dim = rhythm_dim

        self.branch_small = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.branch_medium = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GELU(),
        )

        self.branch_large = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(32 * 3, 64, kernel_size=1),
            nn.GELU(),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(64, rhythm_dim),
            nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(rhythm_dim)

    def forward(self, frame_scores, video_mask=None):
        """
        Args:
            frame_scores: [B, T] 帧级美学分数
            video_mask: [B, T] 视频掩码

        Returns:
            rhythm_features: [B, rhythm_dim] 节奏模式特征
        """
        score_sequence = frame_scores.unsqueeze(1)  # [B, 1, T]

        feat_small = self.branch_small(score_sequence)
        feat_medium = self.branch_medium(score_sequence)
        feat_large = self.branch_large(score_sequence)

        multi_scale = torch.cat([feat_small, feat_medium, feat_large], dim=1)  # [B, 96, T]
        fused = self.fusion(multi_scale)  # [B, 64, T]

        if video_mask is not None:
            mask_expanded = video_mask.unsqueeze(1).float()
            fused = fused * mask_expanded
            mask_sum = mask_expanded.sum(dim=-1, keepdim=True).clamp(min=1.0)
            pooled = fused.sum(dim=-1) / mask_sum.squeeze(-1)
        else:
            pooled = fused.mean(dim=-1)

        rhythm_features = self.output_proj(pooled)
        rhythm_features = self.output_norm(rhythm_features)

        return rhythm_features


class AestheticRhythmEncoder(nn.Module):
    """
    消融版美学节奏编码器：仅保留 1D CNN 节奏模式特征，移除曲线统计特征。

    原版 output_dim = 10 (统计) + 64 (CNN) = 74
    消融版 output_dim = 64 (仅 CNN)
    """

    def __init__(self, end_ratio=0.25, max_seq_len=32, rhythm_dim=64):
        super().__init__()
        self.rhythm_pattern_encoder = RhythmPatternEncoder(
            max_seq_len=max_seq_len, rhythm_dim=rhythm_dim
        )
        self.output_dim = rhythm_dim  # 64，不再包含统计特征的 10 维

    def forward(self, frame_scores, video_mask=None):
        """
        Args:
            frame_scores: [B, T] 帧级美学分数
            video_mask: [B, T] 视频掩码

        Returns:
            rhythm_features: [B, output_dim] 节奏特征（仅 CNN）
            curve_stats: None（消融掉了，保留接口兼容性）
        """
        rhythm_features = self.rhythm_pattern_encoder(frame_scores, video_mask)
        return rhythm_features, None
