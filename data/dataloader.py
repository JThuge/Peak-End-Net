"""
Peak-End Net Data Loader.

Features:
1. Loads frame-level aesthetic distribution pseudo-labels (generated offline by teacher model)
2. Supports noise-augmented data loading
3. Supports loading from pre-extracted frame cache (avoids re-decoding video each epoch)
4. Includes DIVIDE dataset loader (for cross-dataset evaluation)
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.rawvideo_util import RawVideoExtractor

logger = logging.getLogger("peak_end_net")


def _worker_init_fn(worker_id):
    """DataLoader worker initialization function."""
    cv2.setNumThreads(1)
    os.environ["OPENCV_THREAD_COUNT"] = "1"


# All score columns
ALL_SCORE_COLUMNS = [
    "score", "composition", "shotsize", "lighting",
    "visualtone", "color", "depthoffield",
    "expression", "movement", "costume", "makeup",
]

GENERAL_SCORE_COLUMNS = ["composition", "shotsize", "lighting", "visualtone", "color", "depthoffield"]
HUMAN_SCORE_COLUMNS = ["expression", "movement", "costume", "makeup"]


class PeakAesDatasetV4(Dataset):
    """
    Peak-End Net Dataset.

    Features:
    1. Loads frame-level aesthetic distribution pseudo-labels
    2. Supports returning raw frame images for noise augmentation
    3. Supports loading cached frames from pre-extracted directory
       (uses cache when available, falls back to video decoding)
    """
    
    def __init__(
        self,
        csv_path,
        video_paths_json,
        frame_pseudo_labels_path=None,
        extracted_frames_dir=None,
        max_frames=12,
        feature_framerate=1,
        image_resolution=224,
        frame_order=0,
        slice_framepos=2,
    ):
        self.data = pd.read_csv(csv_path)
        
        with open(video_paths_json, "r") as f:
            self.video_paths = json.load(f)
        
        self.max_frames = max_frames
        self.frame_order = frame_order
        self.slice_framepos = slice_framepos
        self.image_resolution = image_resolution

        # Pre-extracted frames directory: load from cache first, avoid re-decoding each epoch
        self.extracted_frames_dir = extracted_frames_dir

        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution
        )
        
        # Ensure video_id is string type
        self.data["video_id"] = self.data["video_id"].astype(str)

        # Filter out samples without video files
        self.data = self.data[
            self.data["video_id"].apply(lambda vid: vid in self.video_paths)
        ].reset_index(drop=True)

        # Process label column
        if "label" in self.data.columns:
            self.data["label"] = self.data["label"].astype(str).str.strip()
            self.data["is_human"] = self.data["label"].str.contains(
                "Character", case=False, na=False
            )
        else:
            self.data["is_human"] = False

        # Process score columns
        for col in ALL_SCORE_COLUMNS:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        # Fill missing values with column mean for general attributes and overall score
        for col in GENERAL_SCORE_COLUMNS + ["score"]:
            if col in self.data.columns:
                col_mean = self.data[col].mean()
                nan_count = self.data[col].isna().sum()
                self.data[col] = self.data[col].fillna(col_mean)
                if nan_count > 0:
                    logger.info(f"  [{col}] filled {nan_count} NaN with mean={col_mean:.2f}")

        # Fill 0 for human attributes when sample is not human category
        for col in HUMAN_SCORE_COLUMNS:
            if col in self.data.columns:
                human_mask = self.data["is_human"]
                if human_mask.any():
                    human_mean = self.data.loc[human_mask, col].mean()
                    self.data.loc[human_mask, col] = self.data.loc[human_mask, col].fillna(human_mean)
                self.data.loc[~human_mask, col] = self.data.loc[~human_mask, col].fillna(0.0)

        # Load frame-level pseudo labels
        self.frame_pseudo_labels = None
        if frame_pseudo_labels_path and os.path.exists(frame_pseudo_labels_path):
            self._load_frame_pseudo_labels(frame_pseudo_labels_path)

        # Statistics
        human_count = self.data["is_human"].sum()
        non_human_count = len(self.data) - human_count
        logger.info(f"[PeakAesDataset] Initialization complete:")
        logger.info(f"  Total samples: {len(self.data)}")
        logger.info(f"  Human category: {human_count}, Non-human: {non_human_count}")
        logger.info(f"  Max frames: {self.max_frames}, Slice mode: {self.slice_framepos}")
        if self.extracted_frames_dir and os.path.isdir(self.extracted_frames_dir):
            cached_count = sum(
                1 for vid in self.data["video_id"]
                if os.path.exists(os.path.join(self.extracted_frames_dir, f"{vid}.npz"))
            )
            logger.info(f"  Extracted frames dir: {self.extracted_frames_dir}")
            logger.info(f"  Cached frames available: {cached_count}/{len(self.data)}")
        else:
            logger.info(f"  Extracted frames dir: Not set (will decode from video each time)")
        if self.frame_pseudo_labels is not None:
            logger.info(f"  Frame pseudo labels loaded: {len(self.frame_pseudo_labels)} videos")
        else:
            logger.info(f"  Frame pseudo labels: Not loaded")

    def _load_frame_pseudo_labels(self, pseudo_labels_path):
        """Load frame-level aesthetic distribution pseudo-labels."""
        try:
            if pseudo_labels_path.endswith(".npz"):
                # numpy compressed format
                data = np.load(pseudo_labels_path, allow_pickle=True)
                self.frame_pseudo_labels = dict(data)
            elif pseudo_labels_path.endswith(".json"):
                # json format
                with open(pseudo_labels_path, "r") as f:
                    self.frame_pseudo_labels = json.load(f)
            else:
                logger.warning(f"Unknown pseudo labels format: {pseudo_labels_path}")
                return
            
            logger.info(f"Loaded frame pseudo labels from {pseudo_labels_path}")
        except Exception as e:
            logger.warning(f"Failed to load frame pseudo labels: {e}")
            self.frame_pseudo_labels = None

    def __len__(self):
        return len(self.data)

    def _load_cached_frames(self, video_id):
        """Load frame data from pre-extracted cache directory."""
        if self.extracted_frames_dir is None:
            return None, None

        cache_path = os.path.join(self.extracted_frames_dir, f"{video_id}.npz")
        if not os.path.exists(cache_path):
            return None, None

        try:
            data = np.load(cache_path)
            frames = data["frames"]  # [max_frames, 1, 3, H, W]
            mask = data["mask"]      # [max_frames]

            # Wrap to format consistent with _get_rawvideo: [1, max_frames, 1, 3, H, W]
            video = np.zeros(
                (1, self.max_frames, 1, 3,
                 self.image_resolution, self.image_resolution),
                dtype=np.float32,
            )
            video_mask = np.zeros((1, self.max_frames), dtype=np.int64)

            valid_frames = min(frames.shape[0], self.max_frames)
            video[0][:valid_frames, ...] = frames[:valid_frames, ...]
            video_mask[0][:valid_frames] = mask[:valid_frames]

            return video, video_mask
        except Exception as error:
            logger.warning(f"Error loading cached frames for {video_id}: {error}")
            # Remove corrupted cache file to re-extract next time
            try:
                os.remove(cache_path)
                logger.info(f"Removed corrupted cache file: {cache_path}")
            except OSError:
                pass
            return None, None

    def _get_rawvideo(self, video_id):
        """
        Load video frame data.

        Loads from pre-extracted cache first; falls back to video decoding if cache is unavailable.
        """
        # Try loading from cache first
        cached_video, cached_mask = self._load_cached_frames(video_id)
        if cached_video is not None:
            return cached_video, cached_mask

        # Fallback: decode from original video file
        video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
        max_video_length = 0
        
        video = np.zeros(
            (1, self.max_frames, 1, 3,
             self.rawVideoExtractor.size, self.rawVideoExtractor.size),
            dtype=np.float32,
        )
        
        video_path = self.video_paths.get(video_id, "")
        if not os.path.exists(video_path):
            return video, video_mask
        
        try:
            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data["video"]
            
            if len(raw_video_data.shape) > 3:
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data)
                
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[: self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(
                            0, raw_video_slice.shape[0] - 1,
                            num=self.max_frames, dtype=int,
                        )
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice
                
                video_slice = self.rawVideoExtractor.process_frame_order(
                    video_slice, frame_order=self.frame_order
                )
                
                slice_len = video_slice.shape[0]
                max_video_length = max(max_video_length, slice_len)
                if slice_len >= 1:
                    video[0][:slice_len, ...] = video_slice
        except Exception as error:
            logger.warning(f"Error loading video {video_id}: {error}")
        
        video_mask[0][:max_video_length] = 1
        
        return video, video_mask
    
    def _get_frame_pseudo_labels(self, video_id, num_frames):
        """Get frame-level aesthetic distribution pseudo-labels."""
        if self.frame_pseudo_labels is None:
            return None
        
        if video_id not in self.frame_pseudo_labels:
            return None
        
        try:
            video_labels = self.frame_pseudo_labels[video_id]
            
            # Process based on storage format
            if isinstance(video_labels, dict):
                # dict format: {"frame_0": [p1, ..., p10], ...}
                distributions = []
                for i in range(num_frames):
                    frame_key = f"frame_{i}"
                    if frame_key in video_labels:
                        distributions.append(video_labels[frame_key])
                    else:
                        # Uniform distribution as default
                        distributions.append([0.1] * 10)
                return np.array(distributions, dtype=np.float32)
            elif isinstance(video_labels, np.ndarray):
                # numpy array format: [T, 10]
                return video_labels[:num_frames]
            else:
                return None
        except Exception as e:
            logger.warning(f"Error loading pseudo labels for {video_id}: {e}")
            return None
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = row["video_id"]
        
        # Get all 11 scores
        scores = np.array(
            [float(row[col]) if col in row.index else 0.0 for col in ALL_SCORE_COLUMNS],
            dtype=np.float32,
        )
        
        # Whether sample has human attribute annotation
        is_human = bool(row["is_human"])

        # Get video frames
        video, video_mask = self._get_rawvideo(video_id)

        # Get frame pseudo labels, pad to [max_frames, 10]
        num_valid_frames = int(video_mask.sum())
        raw_pseudo_labels = self._get_frame_pseudo_labels(video_id, num_valid_frames)
        
        frame_pseudo_labels = np.zeros((self.max_frames, 10), dtype=np.float32)
        if raw_pseudo_labels is not None and len(raw_pseudo_labels) > 0:
            valid_label_count = min(len(raw_pseudo_labels), self.max_frames)
            frame_pseudo_labels[:valid_label_count] = raw_pseudo_labels[:valid_label_count]
        
        return (
            torch.tensor(video),
            torch.tensor(video_mask),
            torch.tensor(scores),
            torch.tensor(is_human),
            torch.tensor(frame_pseudo_labels),
            video_id,
        )


def create_peakaes_v4_dataloader(csv_path, video_paths_json, args, is_train=True):
    """Create Peak-End Net data loader."""
    dataset = PeakAesDatasetV4(
        csv_path=csv_path,
        video_paths_json=video_paths_json,
        frame_pseudo_labels_path=getattr(args, "frame_pseudo_labels_path", None),
        extracted_frames_dir=getattr(args, "extracted_frames_dir", None),
        max_frames=args.max_frames,
        feature_framerate=args.feature_framerate,
        image_resolution=224,
        frame_order=args.train_frame_order if is_train else args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    
    batch_size = args.batch_size_train if is_train else args.batch_size_val
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_thread_reader,
        shuffle=is_train,
        drop_last=is_train,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        persistent_workers=args.num_thread_reader > 0,
    )
    return dataloader, len(dataset)


# ============================================================
# DIVIDE Dataset (for cross-dataset evaluation)
# ============================================================

class DIVIDEDataset(Dataset):
    """
    DIVIDE dataset loader.

    Label file format per line: "video_name, aesthetic, technical, overall"
    Three scores correspond to aesthetic, technical, and overall quality.
    Evaluation uses the first column (aesthetic).

    Supports loading from pre-extracted frame cache to avoid re-decoding video each evaluation.
    """

    def __init__(
        self,
        label_path,
        video_dir,
        extracted_frames_dir=None,
        max_frames=12,
        feature_framerate=1,
        image_resolution=224,
        frame_order=0,
        slice_framepos=2,
    ):
        self.video_dir = video_dir
        self.max_frames = max_frames
        self.frame_order = frame_order
        self.slice_framepos = slice_framepos
        self.image_resolution = image_resolution
        self.extracted_frames_dir = extracted_frames_dir

        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution
        )

        self.samples = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                video_name = parts[0]
                video_path = os.path.join(video_dir, video_name)
                if not os.path.exists(video_path):
                    continue
                aesthetic_score = float(parts[1])
                self.samples.append({
                    "video_name": video_name,
                    "video_id": os.path.splitext(video_name)[0],
                    "video_path": video_path,
                    "aesthetic_score": aesthetic_score,
                })

        logger.info(f"[DIVIDEDataset] Loaded {len(self.samples)} samples from {label_path}")
        if self.extracted_frames_dir and os.path.isdir(self.extracted_frames_dir):
            cached_count = sum(
                1 for s in self.samples
                if os.path.exists(os.path.join(self.extracted_frames_dir, f"{s['video_id']}.npz"))
            )
            logger.info(f"  Cached frames available: {cached_count}/{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _load_cached_frames(self, video_id):
        """Load frame data from pre-extracted cache directory."""
        if self.extracted_frames_dir is None:
            return None, None

        cache_path = os.path.join(self.extracted_frames_dir, f"{video_id}.npz")
        if not os.path.exists(cache_path):
            return None, None

        try:
            data = np.load(cache_path)
            frames = data["frames"]
            mask = data["mask"]

            video = np.zeros(
                (1, self.max_frames, 1, 3,
                 self.image_resolution, self.image_resolution),
                dtype=np.float32,
            )
            video_mask = np.zeros((1, self.max_frames), dtype=np.int64)

            valid_frames = min(frames.shape[0], self.max_frames)
            video[0][:valid_frames, ...] = frames[:valid_frames, ...]
            video_mask[0][:valid_frames] = mask[:valid_frames]

            return video, video_mask
        except Exception as error:
            logger.warning(f"Error loading cached frames for {video_id}: {error}")
            return None, None

    def _get_rawvideo(self, video_path, video_id):
        """Load video frames, preferring cache when available."""
        # Try loading from cache first
        cached_video, cached_mask = self._load_cached_frames(video_id)
        if cached_video is not None:
            return cached_video, cached_mask

        # Fallback: decode from original video file
        video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
        max_video_length = 0
        video = np.zeros(
            (1, self.max_frames, 1, 3,
             self.rawVideoExtractor.size, self.rawVideoExtractor.size),
            dtype=np.float32,
        )

        if not os.path.exists(video_path):
            return video, video_mask

        try:
            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data["video"]

            if len(raw_video_data.shape) > 3:
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data)

                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(
                            0, raw_video_slice.shape[0] - 1,
                            num=self.max_frames, dtype=int,
                        )
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(
                    video_slice, frame_order=self.frame_order
                )

                slice_len = video_slice.shape[0]
                max_video_length = max(max_video_length, slice_len)
                if slice_len >= 1:
                    video[0][:slice_len, ...] = video_slice
        except Exception:
            pass

        video_mask[0][:max_video_length] = 1
        return video, video_mask

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video, video_mask = self._get_rawvideo(sample["video_path"], sample["video_id"])
        aesthetic_score = np.float32(sample["aesthetic_score"])
        return (
            torch.tensor(video),
            torch.tensor(video_mask),
            torch.tensor(aesthetic_score),
            sample["video_name"],
        )


def create_divide_dataloader(args):
    """Create DIVIDE dataset DataLoader (for cross-dataset evaluation)."""
    divide_root = getattr(args, "divide_root", None)
    if not divide_root or not os.path.isdir(divide_root):
        return None, 0

    label_file = getattr(args, "divide_label_file", "val_labels.txt")
    label_path = os.path.join(divide_root, label_file)
    video_dir = os.path.join(divide_root, "videos")

    if not os.path.exists(label_path):
        logger.warning(f"DIVIDE label file not found: {label_path}")
        return None, 0

    dataset = DIVIDEDataset(
        label_path=label_path,
        video_dir=video_dir,
        extracted_frames_dir=getattr(args, "divide_extracted_frames_dir", None),
        max_frames=args.max_frames,
        feature_framerate=args.feature_framerate,
        image_resolution=224,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        persistent_workers=args.num_thread_reader > 0,
    )
    return dataloader, len(dataset)
