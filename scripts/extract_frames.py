"""
Video Frame Extraction Script.

Pre-extracts video frames and saves them as numpy files to avoid
re-decoding videos each epoch during training, significantly speeding up data loading.

Storage format:
  Each video is saved as a single .npz file:
  - frames: [T, 1, 3, 224, 224] frame data (float32, normalized)
  - mask: [T] valid frame mask (int64)
  - num_frames: number of valid frames

Usage:
  # Extract VADB dataset frames
  python scripts/extract_frames.py \
      --video_paths_json /path/to/video_paths.json \
      --output_dir /path/to/extracted_frames_vadb \
      --max_frames 12

  # Extract DIVIDE dataset frames
  python scripts/extract_frames.py \
      --divide_label_path /path/to/val_labels.txt \
      --divide_video_dir /path/to/videos \
      --output_dir /path/to/extracted_frames_divide \
      --max_frames 12
"""

import os
import sys
import json
import argparse
import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import numpy as np
import cv2

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.rawvideo_util import RawVideoExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("extract_frames")


def _extract_single_video(task, output_dir, max_frames, feature_framerate,
                          image_resolution, slice_framepos, frame_order):
    """
    Worker function: extract frames from a single video and save as .npz file.

    Creates a separate RawVideoExtractor instance inside each worker process
    (not shared across processes). Uses _extract_single_video._extractor cache
    to avoid re-creating the extractor for every video.

    Args:
        task: (video_id, video_path) tuple
        Other args: frame extraction configuration

    Returns:
        (status, video_id)
        status: "success" / "skipped" / "failed"
    """
    video_id, video_path = task
    output_path = os.path.join(output_dir, f"{video_id}.npz")

    # Skip if already extracted
    if os.path.exists(output_path):
        return "skipped", video_id

    if not os.path.exists(video_path):
        return "failed", video_id

    # Cache RawVideoExtractor instance per worker process
    if not hasattr(_extract_single_video, "_extractor"):
        cv2.setNumThreads(1)
        os.environ["OPENCV_THREAD_COUNT"] = "1"
        _extract_single_video._extractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution
        )
    extractor = _extract_single_video._extractor

    try:
        raw_video_data = extractor.get_video_data(video_path)
        raw_video_data = raw_video_data["video"]

        if len(raw_video_data.shape) <= 3:
            return "failed", video_id

        raw_video_slice = extractor.process_raw_data(raw_video_data)

        if max_frames < raw_video_slice.shape[0]:
            if slice_framepos == 0:
                video_slice = raw_video_slice[:max_frames, ...]
            elif slice_framepos == 1:
                video_slice = raw_video_slice[-max_frames:, ...]
            else:
                sample_indx = np.linspace(
                    0, raw_video_slice.shape[0] - 1,
                    num=max_frames, dtype=int,
                )
                video_slice = raw_video_slice[sample_indx, ...]
        else:
            video_slice = raw_video_slice

        video_slice = extractor.process_frame_order(
            video_slice, frame_order=frame_order
        )

        num_frames = video_slice.shape[0]

        # Build format consistent with dataloader: [max_frames, 1, 3, H, W]
        frames = np.zeros(
            (max_frames, 1, 3, extractor.size, extractor.size),
            dtype=np.float32,
        )
        mask = np.zeros(max_frames, dtype=np.int64)

        frames[:num_frames, ...] = video_slice
        mask[:num_frames] = 1

        # Save compressed
        np.savez_compressed(
            output_path,
            frames=frames,
            mask=mask,
            num_frames=num_frames,
        )

        return "success", video_id

    except Exception:
        return "failed", video_id


def _run_parallel_extraction(tasks, output_dir, max_frames, feature_framerate,
                             image_resolution, slice_framepos, frame_order,
                             num_workers, description):
    """
    Run parallel frame extraction using multiprocessing.

    Args:
        tasks: [(video_id, video_path), ...] list
        output_dir: output directory
        num_workers: number of parallel workers
        description: progress bar description
    """
    os.makedirs(output_dir, exist_ok=True)

    worker_fn = partial(
        _extract_single_video,
        output_dir=output_dir,
        max_frames=max_frames,
        feature_framerate=feature_framerate,
        image_resolution=image_resolution,
        slice_framepos=slice_framepos,
        frame_order=frame_order,
    )

    success_count = 0
    skip_count = 0
    fail_count = 0

    with Pool(processes=num_workers) as pool:
        for status, video_id in tqdm(
            pool.imap_unordered(worker_fn, tasks),
            total=len(tasks),
            desc=description,
        ):
            if status == "skipped":
                skip_count += 1
            elif status == "success":
                success_count += 1
            else:
                fail_count += 1

    logger.info(f"{description} complete: {success_count} success, "
                f"{skip_count} skipped, {fail_count} failed")


def extract_vadb_frames(video_paths_json, output_dir, max_frames, feature_framerate,
                        image_resolution, slice_framepos, frame_order, num_workers=8):
    """Extract frames from all VADB videos using multiprocessing."""
    with open(video_paths_json, "r") as f:
        video_paths = json.load(f)

    logger.info(f"VADB: {len(video_paths)} videos to process")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Using {num_workers} workers")

    tasks = list(video_paths.items())

    _run_parallel_extraction(
        tasks=tasks,
        output_dir=output_dir,
        max_frames=max_frames,
        feature_framerate=feature_framerate,
        image_resolution=image_resolution,
        slice_framepos=slice_framepos,
        frame_order=frame_order,
        num_workers=num_workers,
        description="Extracting VADB frames",
    )


def extract_divide_frames(label_path, video_dir, output_dir, max_frames,
                          feature_framerate, image_resolution, slice_framepos,
                          frame_order, num_workers=8):
    """Extract frames from all DIVIDE videos using multiprocessing."""
    video_list = []
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
            if os.path.exists(video_path):
                video_id = os.path.splitext(video_name)[0]
                video_list.append((video_id, video_path))

    logger.info(f"DIVIDE: {len(video_list)} videos to process")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Using {num_workers} workers")

    _run_parallel_extraction(
        tasks=video_list,
        output_dir=output_dir,
        max_frames=max_frames,
        feature_framerate=feature_framerate,
        image_resolution=image_resolution,
        slice_framepos=slice_framepos,
        frame_order=frame_order,
        num_workers=num_workers,
        description="Extracting DIVIDE frames",
    )


def main():
    parser = argparse.ArgumentParser(description="Extract and cache video frames")

    # VADB arguments
    parser.add_argument("--video_paths_json", type=str, default=None,
                        help="Path to VADB video_paths.json")

    # DIVIDE arguments
    parser.add_argument("--divide_label_path", type=str, default=None,
                        help="Path to DIVIDE label file (e.g., val_labels.txt)")
    parser.add_argument("--divide_video_dir", type=str, default=None,
                        help="Path to DIVIDE video directory")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for extracted frames")

    # Frame extraction parameters
    parser.add_argument("--max_frames", type=int, default=12)
    parser.add_argument("--feature_framerate", type=int, default=1)
    parser.add_argument("--image_resolution", type=int, default=224)
    parser.add_argument("--slice_framepos", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--frame_order", type=int, default=0, choices=[0, 1, 2])

    # Parallelism parameters
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel workers for frame extraction")

    args = parser.parse_args()

    actual_workers = min(args.num_workers, cpu_count())
    logger.info(f"Parallel workers: {actual_workers} (requested: {args.num_workers}, "
                f"available CPUs: {cpu_count()})")

    if args.video_paths_json:
        extract_vadb_frames(
            video_paths_json=args.video_paths_json,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            feature_framerate=args.feature_framerate,
            image_resolution=args.image_resolution,
            slice_framepos=args.slice_framepos,
            frame_order=args.frame_order,
            num_workers=actual_workers,
        )

    if args.divide_label_path and args.divide_video_dir:
        extract_divide_frames(
            label_path=args.divide_label_path,
            video_dir=args.divide_video_dir,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            feature_framerate=args.feature_framerate,
            image_resolution=args.image_resolution,
            slice_framepos=args.slice_framepos,
            frame_order=args.frame_order,
            num_workers=actual_workers,
        )

    if not args.video_paths_json and not args.divide_label_path:
        logger.error("Please specify --video_paths_json (VADB) or "
                      "--divide_label_path + --divide_video_dir (DIVIDE)")


if __name__ == "__main__":
    main()
