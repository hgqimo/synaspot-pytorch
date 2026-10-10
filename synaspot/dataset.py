"""SYNASPOT 训练数据加载.

清单格式 (04_build_dataset.py 输出): 每行含
    key, wav, split, speaker, spk_id, text, pinyin, label(npy), num_frames, duration
FBank 用 kaldi 参数 (25ms/10ms) 提取, 与帧标签严格对齐:
    num_frames = (num_samples - 400) // 160 + 1
"""
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.compliance import kaldi

from .config import ModelConfig


def load_wav(path: str, target_sr: int) -> torch.Tensor:
    """读取音频为 (1, T) float32 波形并重采样到 target_sr."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
    wav = torch.from_numpy(data.T)  # (C, T)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


# 各音频主干的原始输出帧率 (重采样到 cfg.feat_frame_rate 之前)
NATIVE_HZ = {"dfsmn": 100, "ssl": 50}


def resample_time(feats: torch.Tensor, native_hz: int, target_hz: int) -> torch.Tensor:
    """把 (T, D) 特征沿时间轴从 native_hz 重采样到 target_hz.

    降采样取每 k 帧一帧, 升采样逐帧重复; 帧率整除关系由配置保证.
    """
    if native_hz == target_hz:
        return feats
    if native_hz > target_hz:
        return feats[::(native_hz // target_hz)]
    return feats.repeat_interleave(target_hz // native_hz, dim=0)


class SynaspotDataset(Dataset):
    """音频主干有两档, 输出统一重采样到 cfg.feat_frame_rate:

    - "dfsmn":     在线提 80 维 fbank(100Hz), 句级减均值.
    - "ssl":       读离线缓存的 HuBERT 特征(768 维, 50Hz), feat_dir 指向 hubert_feats/{split}.

    帧标签恒为 100Hz, 按 cfg.label_ds = 100/feat_frame_rate 降采样对齐特征.
    """

    def __init__(self, manifest_path: str, cfg: ModelConfig,
                 max_frames: int = 1600, training: bool = True,
                 limit: int = 0, dither: bool = None, feat_dir: str = None,
                 teacher_feat_dir: str = None, teacher_backbone: str = None):
        self.cfg = cfg
        self.training = training
        self.max_frames = max_frames  # 100Hz 口径的帧数上限(按时长过滤)
        self.backbone = cfg.audio_backbone
        # ssl 需离线缓存目录; dfsmn 在线提 fbank, 无需
        self.feat_dir = Path(feat_dir) if feat_dir else None
        # 蒸馏: 教师(离线主干)特征缓存目录与主干. 非 None 时 __getitem__ 额外返回
        # 教师特征, 供冻结教师前向. 教师/学生特征都重采样到 feat_frame_rate 后逐帧对齐.
        self.teacher_feat_dir = Path(teacher_feat_dir) if teacher_feat_dir else None
        self.teacher_backbone = teacher_backbone
        # 100Hz 帧标签 -> feat_frame_rate 的降采样倍数
        self.ds = cfg.label_ds
        # feat_frame_rate 口径下的帧数上限
        self.max_frames_feat = max_frames // self.ds
        # dither 默认随 training; 过拟合自检时应显式关掉, 否则特征噪声妨碍拟合判读
        self.dither = dither if dither is not None else training
        self.records = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["num_frames"] <= max_frames:
                    self.records.append(rec)
        # limit>0: 只取前 N 条, 用于小子集过拟合自检(判别架构/标签是否有 bug)
        if limit > 0:
            self.records = self.records[:limit]
        self.sizes = [r["num_frames"] // self.ds for r in self.records]

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _load_cached(rec, feat_dir: Path, backbone: str) -> tuple:
        """读离线缓存特征 (ssl), 返回 (feats (T,D), 原生帧率)."""
        feats = torch.from_numpy(
            np.load(feat_dir / f"{rec['key']}.npy").astype(np.float32))
        return feats, NATIVE_HZ[backbone]

    def _load_feats(self, rec) -> tuple:
        """按学生主干载入特征, 返回 (feats (T,D), 原生帧率)."""
        if self.backbone == "ssl":
            return self._load_cached(rec, self.feat_dir, self.backbone)
        # dfsmn: 在线 fbank (100Hz)
        wav = load_wav(rec["wav"], self.cfg.sample_rate)
        feats = kaldi.fbank(
            wav, num_mel_bins=self.cfg.num_fbanks,
            frame_length=self.cfg.frame_length_ms,
            frame_shift=self.cfg.frame_shift_ms,
            sample_frequency=self.cfg.sample_rate,
            dither=1.0 if self.dither else 0.0,
        )
        if self.cfg.use_delta:
            from torchaudio.functional import compute_deltas
            d = compute_deltas(feats)
            dd = compute_deltas(d)
            feats = torch.cat([feats, d, dd], dim=-1)
        return feats - feats.mean(dim=0), 100  # 句级均值归一化

    def _load_teacher_feats(self, rec) -> tuple:
        """按教师主干载入离线缓存特征 (蒸馏用), 返回 (feats (T,D), 原生帧率)."""
        return self._load_cached(rec, self.teacher_feat_dir, self.teacher_backbone)

    def __getitem__(self, idx):
        rec = self.records[idx]
        labels = torch.from_numpy(np.load(rec["label"]).astype(np.int64))
        labels = labels[::self.ds]  # 100Hz -> feat_frame_rate
        feats, native_hz = self._load_feats(rec)
        feats = resample_time(feats, native_hz, self.cfg.feat_frame_rate)
        if self.teacher_feat_dir is not None:
            tfeats, t_hz = self._load_teacher_feats(rec)
            tfeats = resample_time(tfeats, t_hz, self.cfg.feat_frame_rate)
            # 学生/教师原生帧数可能差 1~2 (取整差异),
            # 统一截到最短, 保证逐帧蒸馏对齐.
            T = min(feats.size(0), tfeats.size(0), labels.size(0),
                    self.max_frames_feat)
            return feats[:T], tfeats[:T], labels[:T], int(rec["spk_id"])
        T = min(feats.size(0), labels.size(0), self.max_frames_feat)
        return feats[:T], None, labels[:T], int(rec["spk_id"])


def collate_fn(batch):
    # __getitem__ 恒返回 (学生特征, 教师特征或None, 标签, spk); 教师特征仅蒸馏时非 None.
    feats, tfeats, labels, spks = zip(*batch)
    bs = len(batch)
    tmax = max(f.size(0) for f in feats)
    fdim = feats[0].size(1)
    fbank = feats[0].new_zeros(bs, tmax, fdim)
    lab = labels[0].new_zeros(bs, tmax)
    mask = torch.zeros(bs, tmax, dtype=torch.bool)
    for i, (f, l) in enumerate(zip(feats, labels)):
        t = f.size(0)
        fbank[i, :t] = f
        lab[i, :t] = l
        mask[i, :t] = True
    out = {
        "fbank": fbank,
        "labels": lab,
        "mask": mask,
        "spk_id": torch.tensor(spks, dtype=torch.long),
    }
    if tfeats[0] is not None:
        # 教师特征与学生特征逐样本等长(已在 __getitem__ 截齐), 共用同一 mask.
        tfdim = tfeats[0].size(1)
        tfbank = tfeats[0].new_zeros(bs, tmax, tfdim)
        for i, tf in enumerate(tfeats):
            tfbank[i, :tf.size(0)] = tf
        out["teacher_fbank"] = tfbank
    return out


class BucketBatchSampler:
    """按长度分桶的批采样器, 减少 padding 浪费; 支持 DDP 分片."""

    def __init__(self, sizes, batch_size: int, shuffle: bool = True,
                 drop_last: bool = False, rank: int = 0, world_size: int = 1,
                 seed: int = 0, bucket_mult: int = 32):
        idx = np.argsort(sizes)[rank::world_size]  # 先按 rank 分片再分桶
        self.buckets = [idx[i:i + batch_size * bucket_mult]
                        for i in range(0, len(idx), batch_size * bucket_mult)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        buckets = self.buckets
        if self.shuffle:
            order = rng.permutation(len(buckets))
        else:
            order = np.arange(len(buckets))
        batches = []
        for bi in order:
            b = list(buckets[bi])
            if self.shuffle:
                rng.shuffle(b)
            for i in range(0, len(b), self.batch_size):
                chunk = b[i:i + self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self):
        n = sum(len(b) for b in self.buckets)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)
