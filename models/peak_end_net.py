"""
Peak-End Net: Peak-End Theory Inspired Video Aesthetic Assessment.

Architecture:
    Input Video [B, 1, T, 1, 3, 224, 224]
        |
        v
    [Frozen CLIP ViT-L/14 Encoder] -> Frame Features [B, T, 768]
        |
        v
    [Frozen AVA Aesthetic Head] -> Frame Distributions [B, T, 10] + Frame Scores [B, T]
        |
        v
    [Key Moment Discovery] -> Attention Weights [B, T]
        |
        v
    [Peak-End Aggregation] -> Video Feature [B, 768]
        |
        |---> [Rhythm Encoder (1D CNN)] -> Rhythm Features [B, 64]
        |
        v
    [Concat: Video Feature + Rhythm Features] -> [B, 832]
        |
        v
    [Scoring Network] -> 11 Aesthetic Scores [B, 11]

The per-frame aesthetic scorer (frozen CLIP ViT-L/14 backbone + frozen AVA head)
is pretrained on AVA and kept frozen. Only Key Moment Discovery, Peak-End
Aggregation, the Rhythm Encoder and the Scoring Network are trained.
"""

import os
import sys
import torch
import torch.nn as nn

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
import modules.module_clip as module_clip

# ============================================================
# Patch module_clip._MODELS to add ViT-L/14 support
# ============================================================
_VIT_L14_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
)
if "ViT-L/14" not in module_clip._MODELS:
    module_clip._MODELS["ViT-L/14"] = _VIT_L14_URL
if "ViT-L/14" not in module_clip._PT_NAME:
    module_clip._PT_NAME["ViT-L/14"] = "ViT-L-14.pt"

# Import model sub-modules from the models package
from models.frame_perceiver import AvaAestheticHead, KeyMomentDiscovery
from models.peak_end_aggregation import PeakEndAggregation
from models.rhythm_encoder import AestheticRhythmEncoder

# Attribute index definitions
NUM_ATTRIBUTES = 11
OVERALL_INDICES = [0]
GENERAL_INDICES = [1, 2, 3, 4, 5, 6]
HUMAN_INDICES = [7, 8, 9, 10]


class ScoringNetwork(nn.Module):
    """Scoring network: predict 11 aesthetic attribute scores from the
    combined feature."""

    def __init__(self, input_dim, hidden_dim=256, num_attributes=NUM_ATTRIBUTES):
        super().__init__()
        self.num_attributes = num_attributes

        # Shared feature backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim),
            nn.GELU(),
        )
        self.backbone_norm = nn.LayerNorm(hidden_dim)

        # Per-attribute scoring heads
        self.score_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_attributes)
        ])

    def forward(self, combined_features):
        """
        Args:
            combined_features: [B, input_dim]

        Returns:
            all_scores: [B, num_attributes]
        """
        shared_features = self.shared_backbone(combined_features)
        shared_features = self.backbone_norm(shared_features)

        all_scores = []
        for head in self.score_heads:
            all_scores.append(head(shared_features))

        all_scores = torch.cat(all_scores, dim=-1)
        return all_scores


def _get_clip_embed_dim(pretrained_clip_name):
    """Return embed_dim for a CLIP model name.

    ViT-B/32, ViT-B/16 -> 512
    ViT-L/14, ViT-L/14@336px -> 768
    """
    embed_dim_map = {
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        "ViT-L/14@336px": 768,
    }
    if pretrained_clip_name in embed_dim_map:
        return embed_dim_map[pretrained_clip_name]
    raise ValueError(
        f"Unknown CLIP model: {pretrained_clip_name}. "
        f"Supported: {list(embed_dim_map.keys())}"
    )


class PeakAesNetV4(nn.Module):
    """Peak-End Net (ViT-L/14)."""

    def __init__(
        self,
        # Encoder parameters
        pretrained_encoder_path=None,
        pretrained_clip_name="ViT-L/14",
        cross_model="cross-base",
        linear_patch="3d",
        sim_header="meanP",
        max_words=32,
        max_frames=12,
        # Frame aesthetic perceiver (AVA head) parameters
        ava_checkpoint_path=None,
        # Key moment / aggregation parameters
        frame_context_window=3,
        end_window=3,
        # Rhythm encoder parameters
        rhythm_dim=64,
        end_ratio=0.25,
        # Scoring network parameters
        scoring_hidden_dim=256,
    ):
        super().__init__()

        self.max_frames = max_frames
        self.max_words = max_words

        # Dynamically resolve embed_dim
        self.embed_dim = _get_clip_embed_dim(pretrained_clip_name)

        # ============================================================
        # Frozen CLIP visual encoder (ViT-L/14)
        # ============================================================
        self.encoder = self._load_encoder(
            pretrained_encoder_path=pretrained_encoder_path,
            pretrained_clip_name=pretrained_clip_name,
            cross_model=cross_model,
            linear_patch=linear_patch,
            sim_header=sim_header,
            max_words=max_words,
            max_frames=max_frames,
        )

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # ============================================================
        # Frozen AVA aesthetic head (per-frame distribution scorer)
        # ============================================================
        self.ava_head = AvaAestheticHead(
            checkpoint_path=ava_checkpoint_path,
            feature_dim=self.embed_dim,
        )

        # ============================================================
        # Key Moment Discovery: per-frame attention weights
        # ============================================================
        self.key_moment_discovery = KeyMomentDiscovery(end_window=end_window)

        # ============================================================
        # Peak-End Aggregation (single-weight weighted pooling + proj)
        # ============================================================
        self.peak_end_aggregation = PeakEndAggregation(
            embed_dim=self.embed_dim,
            output_dim=self.embed_dim,
        )

        # ============================================================
        # Aesthetic Rhythm Encoder (1D CNN, 64-dim)
        # ============================================================
        self.rhythm_encoder = AestheticRhythmEncoder(
            end_ratio=end_ratio,
            max_seq_len=max_frames,
            rhythm_dim=rhythm_dim,
        )

        # ============================================================
        # Scoring network (input = video feature + rhythm feature)
        # ============================================================
        scoring_input_dim = self.embed_dim + self.rhythm_encoder.output_dim
        self.scoring_network = ScoringNetwork(
            input_dim=scoring_input_dim,
            hidden_dim=scoring_hidden_dim,
            num_attributes=NUM_ATTRIBUTES,
        )

    def _load_encoder(
        self,
        pretrained_encoder_path,
        pretrained_clip_name,
        cross_model,
        linear_patch,
        sim_header,
        max_words,
        max_frames,
    ):
        """Load the CLIP4Clip encoder (frozen ViT-L/14)."""

        class TaskConfig:
            pass

        task_config = TaskConfig()
        task_config.pretrained_clip_name = pretrained_clip_name
        task_config.cross_model = cross_model
        task_config.linear_patch = linear_patch
        task_config.sim_header = sim_header
        task_config.max_words = max_words
        task_config.max_frames = max_frames
        task_config.local_rank = 0
        task_config.loose_type = False
        task_config.freeze_layer_num = 0
        task_config.slice_framepos = 2
        task_config.train_frame_order = 0

        if pretrained_encoder_path and os.path.exists(pretrained_encoder_path):
            model_state_dict = torch.load(
                pretrained_encoder_path, map_location="cpu", weights_only=False
            )
        else:
            model_state_dict = None

        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "local")
        encoder = CLIP4Clip.from_pretrained(
            cross_model,
            cache_dir=cache_dir,
            state_dict=model_state_dict,
            task_config=task_config,
        )

        return encoder

    def extract_frame_features(self, video, video_mask):
        """Extract frame-level features using the frozen encoder."""
        with torch.no_grad():
            frame_features = self.encoder.get_visual_output(video, video_mask)
        return frame_features.float()

    def forward(self, video, video_mask, has_human_labels=None):
        """
        Args:
            video: [B, 1, T, 1, 3, H, W] video tensor
            video_mask: [B, T] video mask
            has_human_labels: [B] bool tensor (unused, kept for interface compat)

        Returns:
            all_scores: [B, 11] predicted scores for all attributes
            auxiliary_outputs: dict with intermediate results
        """
        batch_size = video.shape[0]

        # Unify video_mask to [B, T]
        video_mask_2d = video_mask.view(batch_size, -1)

        # Step 1: frame-level features (frozen encoder)
        frame_features = self.extract_frame_features(video, video_mask)

        # Step 2: frozen AVA head -> per-frame distribution + score
        with torch.no_grad():
            distributions, frame_scores = self.ava_head(frame_features)

        # Step 3: key moment discovery -> per-frame attention weights
        attention_weights, moment_info = self.key_moment_discovery(
            frame_scores, video_mask_2d
        )

        # Step 4: Peak-End Aggregation (weighted pooling + proj)
        video_feature = self.peak_end_aggregation(frame_features, attention_weights)

        # Step 5: aesthetic rhythm encoding (1D CNN)
        rhythm_features, _ = self.rhythm_encoder(frame_scores, video_mask_2d)

        # Step 6: concat + scoring
        combined_features = torch.cat([video_feature, rhythm_features], dim=-1)
        all_scores = self.scoring_network(combined_features)

        auxiliary_outputs = {
            "distributions": distributions,
            "frame_scores": frame_scores,
            "attention_weights": attention_weights,
            "moment_info": moment_info,
        }

        return all_scores, auxiliary_outputs

    def get_trainable_params(self):
        """Return all trainable parameters (name, param)."""
        return [
            (name, param)
            for name, param in self.named_parameters()
            if param.requires_grad
        ]

    def get_param_groups(self, base_lr=1e-3):
        """Grouped parameters for differentiated learning rates."""
        rhythm_params = []
        main_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "rhythm_encoder" in name:
                rhythm_params.append(param)
            else:
                main_params.append(param)

        param_groups = [
            {"params": main_params, "lr": base_lr},
            {"params": rhythm_params, "lr": base_lr * 0.5},
        ]

        return param_groups

    def train(self, mode=True):
        """Override train to keep the encoder and AVA head in eval mode."""
        super().train(mode)
        self.encoder.eval()
        self.ava_head.eval()
        return self

    def print_model_info(self):
        """Print model parameter information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print("=" * 60)
        print(f"Peak-End Net Model Info (embed_dim={self.embed_dim})")
        print("=" * 60)
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters:    {frozen_params:,}")
        print()

        module_stats = {
            "AVA Head (frozen)": self.ava_head,
            "Key Moment Discovery": self.key_moment_discovery,
            "Peak-End Aggregation": self.peak_end_aggregation,
            "Rhythm Encoder": self.rhythm_encoder,
            "Scoring Network": self.scoring_network,
        }
        for name, module in module_stats.items():
            module_total = sum(p.numel() for p in module.parameters())
            module_trainable = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            print(f"  [{name}] total={module_total:,}, trainable={module_trainable:,}")

        print("=" * 60)


class PeakAesLossV4(nn.Module):
    """
    Peak-End Net joint training loss.

    L_total = L_mse(overall) + L_mse(general) + L_mse(human)
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, predicted_scores, target_scores, has_human_labels):
        """
        Args:
            predicted_scores: [B, 11]
            target_scores: [B, 11]
            has_human_labels: [B] bool tensor

        Returns:
            total_loss: scalar
            loss_dict: dict of loss components
        """
        # Overall MSE
        overall_loss = self.mse_loss(
            predicted_scores[:, OVERALL_INDICES[0]],
            target_scores[:, OVERALL_INDICES[0]],
        ).mean()

        # General MSE
        general_loss = self.mse_loss(
            predicted_scores[:, GENERAL_INDICES],
            target_scores[:, GENERAL_INDICES],
        ).mean()

        # Human MSE (only for samples with human annotations)
        if has_human_labels.any():
            human_mask = has_human_labels.unsqueeze(-1).expand(
                -1, len(HUMAN_INDICES)
            ).float()
            human_mse = self.mse_loss(
                predicted_scores[:, HUMAN_INDICES],
                target_scores[:, HUMAN_INDICES],
            )
            human_loss = (human_mse * human_mask).sum() / human_mask.sum().clamp(min=1.0)
        else:
            human_loss = torch.tensor(0.0, device=predicted_scores.device)

        total_loss = overall_loss + general_loss + human_loss

        loss_dict = {
            "overall_mse": overall_loss.item(),
            "general_mse": general_loss.item(),
            "human_mse": human_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict
