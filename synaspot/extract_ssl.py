"""离线提取预训练 SSL 音频特征 (HuBERT / wav2vec2), 供训练时冻结使用.

为什么离线缓存:
- SSL 模型 ~94M, 每个训练步在线前向会拖慢训练 5~10 倍; 一次性提好缓存成 .npy,
  训练时直接读, 快且省显存.
- 线上部署只跑小学生模型, SSL 大模型仅在此离线阶段出现, 不破坏轻量部署.

输出: {out_dir}/{key}.npy, float16, 形状 (T_ssl, 768), T_ssl = 帧率 50Hz.
帧率对齐: SSL 50Hz 恰为 fbank 100Hz 的 1/2, 训练侧标签按 [::2] 降采样对齐.

用法:
    python -m synaspot.extract_ssl --split train --batch-size 24
    python -m synaspot.extract_ssl --split dev  --batch-size 24
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .dataset import load_wav

DATA_DIR = Path("dataset/AISHELL-1/processed/synaspot")

def out_len(n_samples: int) -> int:
    """HuBERT 卷积特征器总步长 320: 输出帧数 = (n - 400)//320 + 1."""
    return (n_samples - 400) // 320 + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--model-dir", default="synaspot/Model_File/chinese-hubert-base")
    ap.add_argument("--out-dir", default=None,
                    help="默认 {data_dir}/hubert_feats/{split}")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-frames", type=int, default=1600,
                    help="跳过超长样本(与训练 max_frames 一致), 0=不跳过")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR / "hubert_feats" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 载入模型 ----
    from transformers import HubertModel
    model = HubertModel.from_pretrained(args.model_dir).eval().to(args.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[ssl] 载入 {args.model_dir} ({n_params:.1f}M)")

    # ---- 读清单, 跳过已提取的(断点续提) ----
    records = []
    with open(DATA_DIR / f"{args.split}.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if args.max_frames and rec["num_frames"] > args.max_frames:
                continue
            if not (out_dir / f"{rec['key']}.npy").exists():
                records.append(rec)
    print(f"[ssl] {args.split}: 待提取 {len(records)} 条 (已存在的跳过)")
    if not records:
        print("[ssl] 无需提取, 退出")
        return

    # 按时长排序分桶, 减少 padding 浪费
    records.sort(key=lambda r: r["num_frames"])
    bs = args.batch_size
    t0, done = time.time(), 0

    for bi in range(0, len(records), bs):
        batch = records[bi:bi + bs]
        wavs, lens = [], []
        for rec in batch:
            wav = load_wav(rec["wav"], 16000).squeeze(0)  # (T,)
            wavs.append(wav)
            lens.append(wav.size(0))
        L = max(lens)
        x = torch.zeros(len(batch), L)
        attn = torch.zeros(len(batch), L, dtype=torch.long)
        for i, wav in enumerate(wavs):
            x[i, :lens[i]] = wav
            attn[i, :lens[i]] = 1
        # 逐句归一化 (与 Wav2Vec2FeatureExtractor do_normalize 一致)
        x = (x - x.mean(dim=1, keepdim=True)) / (
            x.std(dim=1, keepdim=True) + 1e-7)
        x = x.to(args.device)
        attn = attn.to(args.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            hs = model(x, attention_mask=attn).last_hidden_state  # (B,T,768)
        hs = hs.float()
        for i, rec in enumerate(batch):
            T = out_len(lens[i])
            feat = hs[i, :T].cpu().numpy().astype(np.float16)
            np.save(out_dir / f"{rec['key']}.npy", feat)
        done += len(batch)
        if (bi // bs) % 20 == 0:
            el = time.time() - t0
            rate = done / max(el, 1e-6)
            eta = (len(records) - done) / max(rate, 1e-6)
            print(f"[ssl] {done}/{len(records)}  {rate:.1f} 条/s  ETA {eta/60:.1f}min",
                  flush=True)

    print(f"[ssl] 完成 {args.split}: {done} 条 -> {out_dir}")


if __name__ == "__main__":
    main()
