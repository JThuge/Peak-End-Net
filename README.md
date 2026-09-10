# 🎬 Peak-End Net

**Peak-End-Net: A Peak-End Rule Inspired Framework for Generalizable Video Aesthetic Assessment**

**Accepted to ACM Multimedia 2026**

[📄 Paper](https://arxiv.org/abs/2607.13941) ·
[🤗 Peak-End-Net](https://huggingface.co/GD-ML/Peak-End-Net/tree/main)

Peak-End Net is a video aesthetic assessment framework inspired by the
**peak-end rule**: people tend to judge an experience disproportionately by its
most salient moments and its ending, rather than by uniformly averaging the
entire experience. Peak-End Net translates this insight into a learnable
temporal model that predicts an overall aesthetic score together with ten
fine-grained attribute scores.

## ✨ Overview

![Peak-End Net pipeline](pipeline.png)

The framework contains five main components:

- **Frame Aesthetic Perceiver** — a frozen CLIP ViT-L/14 encoder and an
  AVA-pretrained aesthetic head produce a 10-bin score distribution and an
  expected aesthetic score for each frame.
- **Key Moment Discovery** — learnable peak-, valley-, and end-aware signals
  are combined into a unified temporal attention distribution.
- **Peak-End Aggregation** — attention-weighted pooling summarizes the frame
  features into a video-level representation.
- **Rhythm Encoder** — a multi-scale 1D CNN with kernel sizes 3, 5, and 7
  captures local fluctuations and longer-range trends in the frame-score
  sequence.
- **Gated Fusion** — a lightweight second-stage module adaptively combines the
  learned video-level score with the mean frame-level AVA score.

## 🛠️ Installation

### Requirements

- Python 3.10 or later
- A recent PyTorch and torchvision build compatible with your CUDA environment
- A CUDA-capable GPU is recommended; multi-GPU training requires Linux with
  CUDA and NCCL. CPU inference is supported but slower

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/AMAP-ML/Peak-End-Net.git
cd Peak-End-Net

conda create -n peak-end-net python=3.10 -y
conda activate peak-end-net
```

Install a matching PyTorch and torchvision build for your CUDA environment by
following the [official instructions](https://pytorch.org/get-started/locally/).
Then install the project dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Pretrained Model and Inference

The self-contained [`Peak-End-Net.pth`](https://huggingface.co/GD-ML/Peak-End-Net/tree/main)
checkpoint includes the CLIP ViT-L/14 encoder, AVA aesthetic head, Peak-End
modules, and gated-fusion module. No separate AVA or Stage 1 checkpoint is
required for inference.

Download the checkpoint:

```bash
hf download GD-ML/Peak-End-Net Peak-End-Net.pth --local-dir ./checkpoints
```

Run inference on a video:

```bash
python inference.py \
    --checkpoint ./checkpoints/Peak-End-Net.pth \
    --video /path/to/video.mp4
```

The script reports the overall score, ten attribute scores, the fusion gate,
and the two scores combined by the gate.

## 📁 Data Preparation

The train/test ID splits used for the VADB and DIVIDE benchmarks are provided
under [`splits/`](splits/).

### Annotation files

Both the training and validation CSV files must contain a `video_id` column and
the following 11 score columns:

```text
score, composition, shotsize, lighting, visualtone, color, depthoffield,
expression, movement, costume, makeup
```

An optional `label` column identifies samples with human-centric annotations.
Rows whose label contains `Character` contribute to the loss for
`expression`, `movement`, `costume`, and `makeup`; without this column, those
four attributes are not supervised.

The video-path JSON file maps each `video_id` to its source video:

```json
{
  "video_001": "/path/to/videos/video_001.mp4",
  "video_002": "/path/to/videos/video_002.mp4"
}
```

The JSON keys must match the `video_id` values in both CSV files.

### Optional frame extraction

Training can decode videos on the fly. For faster data loading, pre-extract and
cache the sampled frames as `.npz` files:

```bash
python scripts/extract_frames.py \
    --video_paths_json /path/to/video_paths.json \
    --output_dir ./data/extracted_frames \
    --max_frames 12 \
    --num_workers 16
```

Pass the cache directory to either training stage with:

```bash
--extracted_frames_dir ./data/extracted_frames
```

## 🏋️ Training

Training consists of three steps: pretrain the frame-level AVA aesthetic head,
train Peak-End Net, and finally train the gated-fusion module.

### 1. Pretrain the AVA aesthetic head

The [AVA dataset](https://github.com/mtobeiyf/ava_downloader) label file is
space-separated, with one row per image:

```text
index image_id count_1 count_2 ... count_10 [extra columns ...]
```

Here, `count_i` is the number of votes for aesthetic score `i` (1–10). Store
each image as `{image_id}.jpg` under `--img_dir`.

The pretraining script uses multi-GPU distributed data parallelism:

```bash
torchrun --nproc_per_node=8 ava_pretrain/train_ava_model.py \
    --img_dir /path/to/AVA/images \
    --csv_file /path/to/AVA.txt \
    --clip_model ViT-L/14 \
    --batch_size 64 \
    --epochs 20 \
    --lr 1e-4 \
    --save_dir ./checkpoints
```

The CLIP backbone remains frozen while the 10-bin aesthetic distribution head
is optimized with Earth Mover's Distance loss. The checkpoint with the highest
validation SROCC is saved to `./checkpoints/best_model.pth`.

### 2. Train Stage 1: Peak-End Net

Stage 1 freezes the CLIP encoder and AVA aesthetic head, then trains Key Moment
Discovery, Peak-End Aggregation, the Rhythm Encoder, and the Scoring Network.

Single-GPU training:

```bash
python train.py \
    --train_csv /path/to/train.csv \
    --val_csv /path/to/val.csv \
    --video_paths_json /path/to/video_paths.json \
    --ava_checkpoint_path ./checkpoints/best_model.pth \
    --output_dir ./output_stage1 \
    --epochs 30 \
    --lr 1e-3 \
    --batch_size_train 16 \
    --batch_size_val 8
```

Multi-GPU training:

```bash
torchrun --nproc_per_node=4 train.py \
    --train_csv /path/to/train.csv \
    --val_csv /path/to/val.csv \
    --video_paths_json /path/to/video_paths.json \
    --ava_checkpoint_path ./checkpoints/best_model.pth \
    --output_dir ./output_stage1 \
    --epochs 30 \
    --lr 1e-3 \
    --batch_size_train 16 \
    --batch_size_val 8
```

In distributed training, the batch-size arguments are applied per GPU. The
best Stage 1 checkpoint is saved as `./output_stage1/peakaes_v4_best.pth`.

### 3. Train Stage 2: Gated Fusion

Stage 2 freezes the complete Stage 1 model and trains only the lightweight
gated-fusion module:

```bash
torchrun --nproc_per_node=4 train_stage2.py \
    --stage1_checkpoint ./output_stage1/peakaes_v4_best.pth \
    --ava_checkpoint_path ./checkpoints/best_model.pth \
    --train_csv /path/to/train.csv \
    --val_csv /path/to/val.csv \
    --video_paths_json /path/to/video_paths.json \
    --output_dir ./output_stage2 \
    --epochs 15 \
    --lr 1e-3 \
    --warmup_epochs 2 \
    --gate_loss_weight 1.0 \
    --batch_size_train 16 \
    --batch_size_val 32
```

For single-GPU training, replace `torchrun --nproc_per_node=4` with `python`.
The best fusion checkpoint is saved as `./output_stage2/stage2_best.pth` and
contains the trainable fusion weights only. It is a training checkpoint and
cannot be passed directly to `inference.py`; inference expects the released
self-contained checkpoint from Hugging Face.

## 📝 Note on Evaluation Metrics

The validation metric on the VADB dataset reported in the paper was incorrectly
labeled as RMSE. The values were actually computed using **MSE** (Mean Squared
Error), as implemented in this repository (`mean_squared_error` in
`train.py`). MSE is used consistently as both the training loss and the
validation metric on VADB's official train/test split, making it a natural
choice for model selection and evaluation. The reported numerical values are
correct; only the metric name was mislabeled. The primary evaluation metrics
of this work are PLCC, SROCC, and KRCC, which remain unchanged. We will update
the arXiv version and request a correction/erratum for the published
proceedings.

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@misc{li2026peakendnetpeakendruleinspired,
      title={Peak-End-Net: A Peak-End Rule Inspired Framework for Generalizable Video Aesthetic Assessment},
      author={Geng Li and Haiwen Li and Rui Chen and Jing Tang and Lei Sun and Xiangxiang Chu},
      year={2026},
      eprint={2607.13941},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.13941},
}
```

## 🤝 Acknowledgments

This project builds on [CLIP](https://github.com/openai/CLIP) and
[CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip).

## 📄 License

This project is released under the [MIT License](LICENSE).
