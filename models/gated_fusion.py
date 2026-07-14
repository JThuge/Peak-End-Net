"""
Stage 2: Gated Fusion for Peak-End Net.

Core idea:
- Stage 1: train Peak-End Net (PeakAesNetV4) to obtain the best VADB checkpoint.
- Stage 2: freeze all Stage-1 parameters and train only a lightweight
  GatedFusionModule that dynamically balances the trust between the Stage-1
  model overall score (S_model) and the average frame-level AVA aesthetic score
  (S_static).
- Final score: S_final = gate * S_model + (1 - gate) * S_static

S_static is the mean per-frame aesthetic score from the same frozen AVA head
(ViT-L/14) used in Stage 1, recovered to the [0, 10] score scale.

The gate is supervised explicitly: for each sample, whichever of the Stage-1
overall score / average frame score is closer to the label receives higher
trust (soft target).
"""

import os
import torch
import torch.nn as nn

from models.peak_end_net import PeakAesNetV4


class GatedFusionModule(nn.Module):
    """
    Gated fusion module: a lightweight MLP that dynamically adjusts the trust
    between the Stage-1 overall score and the average frame-level AVA score
    based on combined_features.

    Design:
    - Input: combined_features [B, input_dim]
    - Output: gate [B, 1] in [0, 1]
    - Initialization: final bias = 2.0 -> sigmoid(2) ~= 0.88, initially trusts
      the Stage-1 model.
    """

    def __init__(self, input_dim=832, hidden_dim=128):
        super().__init__()

        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize the gate toward a high value (trust the Stage-1 model).
        nn.init.constant_(self.gate_network[-1].bias, 2.0)  # sigmoid(2) ~= 0.88

        for module in self.gate_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None and module is not self.gate_network[-1]:
                    nn.init.zeros_(module.bias)

    def forward(self, combined_features):
        """
        Args:
            combined_features: [B, input_dim] concatenated video + rhythm features

        Returns:
            gate: [B, 1] gate value in [0, 1]
        """
        gate = torch.sigmoid(self.gate_network(combined_features))
        return gate


class GatedFusionPeakAesNetV4(nn.Module):
    """
    Gated Fusion Peak-End Net: wraps a frozen Stage-1 model + a trainable
    fusion module.

    1. Stage-1 model: frozen, loaded from the Stage-1 checkpoint.
    2. Fusion module: the only trainable part.
    3. Final score: S_final = gate * S_model + (1 - gate) * S_static, where
       S_static is the average frame-level AVA score from the frozen Stage-1
       AVA head.
    """

    def __init__(
        self,
        stage1_checkpoint_path=None,
        # Stage-1 model parameters (must match Stage 1)
        pretrained_encoder_path=None,
        ava_checkpoint_path=None,
        pretrained_clip_name="ViT-L/14",
        cross_model="cross-base",
        linear_patch="3d",
        sim_header="meanP",
        max_words=32,
        max_frames=12,
        frame_context_window=3,
        end_window=3,
        rhythm_dim=64,
        end_ratio=0.25,
        scoring_hidden_dim=256,
        # Fusion module parameters
        fusion_hidden_dim=128,
    ):
        super().__init__()

        # 1. Build the Stage-1 model
        self.v4_model = PeakAesNetV4(
            pretrained_encoder_path=pretrained_encoder_path,
            ava_checkpoint_path=ava_checkpoint_path,
            pretrained_clip_name=pretrained_clip_name,
            cross_model=cross_model,
            linear_patch=linear_patch,
            sim_header=sim_header,
            max_words=max_words,
            max_frames=max_frames,
            frame_context_window=frame_context_window,
            end_window=end_window,
            rhythm_dim=rhythm_dim,
            end_ratio=end_ratio,
            scoring_hidden_dim=scoring_hidden_dim,
        )

        # 2. Load the Stage-1 checkpoint and freeze all parameters.
        #    When stage1_checkpoint_path is None (e.g. loading a merged self-contained
        #    checkpoint later via load_state_dict), skip file loading and just
        #    freeze the Stage-1 model.
        if stage1_checkpoint_path is not None:
            self._load_and_freeze_stage1(stage1_checkpoint_path)
        else:
            for param in self.v4_model.parameters():
                param.requires_grad = False
            self.v4_model.eval()

        # 3. Fusion module (the only trainable part)
        # combined_features dim: embed_dim(768) + rhythm output_dim(64) = 832
        combined_dim = self.v4_model.embed_dim + self.v4_model.rhythm_encoder.output_dim
        self.fusion_module = GatedFusionModule(
            input_dim=combined_dim,
            hidden_dim=fusion_hidden_dim,
        )

        print("[GatedFusionPeakAesNetV4] Initialized:")
        print(f"  Stage-1 model: frozen, loaded from {stage1_checkpoint_path}")
        print(f"  Fusion module: trainable, input_dim={combined_dim}, hidden_dim={fusion_hidden_dim}")
        print(f"  Trainable parameters: {sum(p.numel() for p in self.fusion_module.parameters()):,}")

    def _load_and_freeze_stage1(self, checkpoint_path):
        """Load the Stage-1 checkpoint and freeze all Stage-1 parameters."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {checkpoint_path}")

        print(f"[GatedFusionPeakAesNetV4] Loading Stage-1 checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "trainable_state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint missing 'trainable_state_dict': {checkpoint_path}")

        trainable_state = checkpoint["trainable_state_dict"]

        current_state = self.v4_model.state_dict()
        loaded_count = 0
        for name, param in trainable_state.items():
            if name in current_state:
                if param.shape != current_state[name].shape:
                    print(f"  Warning: Shape mismatch for {name}: {param.shape} vs {current_state[name].shape}")
                    continue
                current_state[name].copy_(param)
                loaded_count += 1
            else:
                print(f"  Warning: Key {name} not found in model state_dict")

        print(f"  Loaded {loaded_count}/{len(trainable_state)} parameters")

        for param in self.v4_model.parameters():
            param.requires_grad = False
        self.v4_model.eval()

        v4_params = sum(p.numel() for p in self.v4_model.parameters())
        print(f"  Frozen {v4_params:,} Stage-1 parameters")

    def forward(self, video, video_mask, has_human_labels=None):
        """
        Args:
            video: [B, 1, T, 1, 3, H, W] video tensor
            video_mask: [B, T] video mask
            has_human_labels: [B] bool tensor (unused, kept for interface compat)

        Returns:
            final_scores: [B, 11] final predicted scores (overall dimension
                          fused, other dimensions unchanged)
            auxiliary_outputs: dict with intermediate results
        """
        batch_size = video.shape[0]
        video_mask_2d = video_mask.view(batch_size, -1)

        with torch.no_grad():
            # Step 1: Stage-1 model prediction
            all_scores, v4_aux = self.v4_model(
                video, video_mask, has_human_labels=has_human_labels,
            )
            S_model = all_scores[:, 0]  # [B] overall dimension

            # Step 2: re-extract combined_features from the Stage-1 model
            frame_features = self.v4_model.extract_frame_features(video, video_mask)
            distributions, frame_scores = self.v4_model.ava_head(frame_features)
            attention_weights, _ = self.v4_model.key_moment_discovery(
                frame_scores, video_mask_2d
            )
            video_feature = self.v4_model.peak_end_aggregation(
                frame_features, attention_weights
            )
            rhythm_features, _ = self.v4_model.rhythm_encoder(frame_scores, video_mask_2d)
            combined_features = torch.cat([video_feature, rhythm_features], dim=-1)

            # Step 3: average frame-level AVA score S_static (same AVA head)
            # ava_head frame_scores are normalized to [0, 1]; recover to [0, 10].
            frame_scores_scaled = frame_scores * 10.0
            valid_counts = video_mask_2d.sum(dim=1).clamp(min=1)  # [B]
            S_static = (frame_scores_scaled * video_mask_2d.float()).sum(dim=1) / valid_counts  # [B]

        # Step 4: gated fusion (trainable)
        gate = self.fusion_module(combined_features)  # [B, 1]

        # Step 5: fuse the overall dimension
        S_final_overall = gate.squeeze(-1) * S_model + (1 - gate.squeeze(-1)) * S_static

        # Step 6: build the final scores (overall dimension fused, others unchanged)
        final_scores = all_scores.clone()
        final_scores[:, 0] = S_final_overall

        auxiliary_outputs = {
            "gate": gate,
            "S_model": S_model,
            "S_static": S_static,
            "combined_features": combined_features,
            "v4_auxiliary": v4_aux,
            "frame_scores": frame_scores,
        }

        return final_scores, auxiliary_outputs

    def train(self, mode=True):
        """Override train to keep the Stage-1 model in eval mode."""
        super().train(mode)
        self.v4_model.eval()
        return self

    def get_trainable_params(self):
        """Get all trainable parameters (only the fusion module)."""
        return [
            (name, param)
            for name, param in self.named_parameters()
            if param.requires_grad
        ]

    def print_model_info(self):
        """Print model parameter information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print("=" * 60)
        print("GatedFusion Peak-End Net Model Info")
        print("=" * 60)
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters:    {frozen_params:,}")
        print()

        fusion_params = sum(p.numel() for p in self.fusion_module.parameters())
        print(f"  [Fusion Module] total={fusion_params:,}, trainable={fusion_params:,}")
        print("  [Stage-1 Model] frozen (loaded from checkpoint)")
        print("=" * 60)
