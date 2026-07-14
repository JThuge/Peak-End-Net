#这段代码的原理在于通过 OpenCV 读取视频帧，然后使用 torchvision 转换帧图像以便能够在 PyTorch 中使用进行后续处理，如视频特征提取、模型训练等用途。
import os
import sys
import ctypes

# 在导入 cv2 之前，先抑制 ffmpeg 底层的 stderr 日志输出
# 这些 warning（如 "Invalid NAL unit size"、"partial file"）来自 ffmpeg C 库，
# 无法通过 Python logging 或 OpenCV API 控制，只能在 C 层面重定向 stderr
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_THREAD_COUNT"] = "1"

import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import cv2

# 限制 OpenCV 线程数，避免多进程分布式训练时线程爆炸
cv2.setNumThreads(1)


def _suppress_ffmpeg_stderr():
    """
    通过 C 层面重定向 stderr 的 fd 到 /dev/null，
    彻底抑制 ffmpeg 底层解码器输出的 warning 日志。
    仅在 Linux 上生效。
    """
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        # 保存原始 stderr fd
        original_stderr_fd = os.dup(2)
        # 将 fd 2 (stderr) 重定向到 /dev/null
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        return original_stderr_fd
    except OSError:
        return None


def _restore_stderr(original_stderr_fd):
    """恢复原始的 stderr"""
    if original_stderr_fd is not None:
        try:
            os.dup2(original_stderr_fd, 2)
            os.close(original_stderr_fd)
        except OSError:
            pass

#RawVideoExtractorCV2 类用于从视频中提取帧并将其转化为 PyTorch 张量
class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)

    #构造用于图像预处理的变换序列，包括调整大小、中心裁剪、转换为 RGB 模式、转换为张量、以及归一化。
    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # 抑制 ffmpeg 底层 C 库的 stderr warning 输出
        saved_stderr = _suppress_ffmpeg_stderr()

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            cap.release()
            _restore_stderr(saved_stderr)
            return {'video': th.zeros(1)}

        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if fps == 0 or frameCount == 0:
            cap.release()
            _restore_stderr(saved_stderr)
            return {'video': th.zeros(1)}

        # 确定采样范围（帧索引）
        if start_time is not None:
            start_frame = int(start_time * fps)
            end_frame = min(int(end_time * fps), frameCount)
        else:
            start_frame = 0
            end_frame = frameCount

        total_frames_in_range = end_frame - start_frame
        if total_frames_in_range <= 0:
            cap.release()
            _restore_stderr(saved_stderr)
            return {'video': th.zeros(1)}

        # 计算需要采样的总帧数：每秒 sample_fp 帧
        if sample_fp > 0:
            total_duration_sec = total_frames_in_range / fps
            target_num_frames = max(1, int(total_duration_sec * sample_fp))
        else:
            target_num_frames = total_frames_in_range

        # 均匀采样帧索引，避免逐帧遍历
        if target_num_frames >= total_frames_in_range:
            frame_indices = list(range(start_frame, end_frame))
        else:
            frame_indices = np.linspace(start_frame, end_frame - 1, num=target_num_frames, dtype=int).tolist()

        # 按顺序读取帧（顺序读取比随机 seek 快得多）
        images = []
        frame_indices_sorted = sorted(set(frame_indices))

        # 使用顺序读取策略：设置到起始帧，然后顺序读取
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices_sorted[0])
        current_frame_pos = frame_indices_sorted[0]
        target_set = set(frame_indices_sorted)

        for target_idx in frame_indices_sorted:
            # 如果需要跳帧，用 seek（仅在跳跃较大时）
            if target_idx > current_frame_pos:
                skip_count = target_idx - current_frame_pos
                if skip_count > fps:
                    # 跳跃较大时用 seek
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                else:
                    # 跳跃较小时顺序跳过（比 seek 快）
                    for _ in range(skip_count):
                        cap.grab()

            ret, frame = cap.read()
            if not ret:
                break
            current_frame_pos = target_idx + 1

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

        cap.release()
        _restore_stderr(saved_stderr)

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data}

    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        return image_input

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based CV2
# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2