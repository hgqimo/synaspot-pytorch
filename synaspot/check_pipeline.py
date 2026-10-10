"""端到端冒烟验证: 真实数据 -> 前向/反向(两阶段损失) -> 保存权重 -> 流式解码自测.

用法:
    python -m synaspot.check_pipeline [--batch-size 8]
"""
import argparse
import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig
from .dataset import SynaspotDataset, collate_fn
from .decode import SynaspotDecoder
from .model import Synaspot
from .train import DEFAULT_DATA, stage_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    with open(data_dir / "dataset_info.json", encoding="utf-8") as f:
        info = json.load(f)
    cfg = ModelConfig(vocab_size=info["vocab_size"],
                      num_speakers=info["num_speakers_train"])

    # ---- 1. 数据加载 ----
    ds = SynaspotDataset(str(data_dir / "train.jsonl"), cfg, max_frames=1000)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    print(f"[1] 数据: batch fbank={tuple(batch['fbank'].shape)} "
          f"labels={tuple(batch['labels'].shape)} spk={batch['spk_id'].tolist()}")

    # ---- 2. 模型 ----
    model = Synaspot(cfg).to(device)
    n_enc = sum(p.numel() for p in model.audio_encoder.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[2] 模型: 编码器 {n_enc / 1e6:.2f}M, 总计 {n_all / 1e6:.2f}M")

    # ---- 3. 阶段一前向/反向 ----
    model.train()
    loss1, logs1 = stage_loss(model, batch, cfg, "audio")
    loss1.backward()
    print(f"[3] stage=audio loss={loss1.item():.4f} "
          + " ".join(f"{k}={v:.4f}" for k, v in logs1.items()))

    # ---- 4. 阶段二前向/反向 ----
    model.zero_grad(set_to_none=True)
    loss2, logs2 = stage_loss(model, batch, cfg, "mixed")
    loss2.backward()
    print(f"[4] stage=mixed loss={loss2.item():.4f} "
          + " ".join(f"{k}={v:.4f}" for k, v in logs2.items()))

    grad_ok = all(p.grad is None or torch.isfinite(p.grad).all()
                  for p in model.parameters())
    print(f"[5] 梯度有限性检查: {'通过' if grad_ok else '失败'}")
    assert grad_ok

    # ---- 5. 保存权重并做流式解码自测 ----
    rec0 = ds.records[0]
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "test.pt"
        with open(data_dir / "pinyin_vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                    "vocab": vocab, "epoch": 0}, ckpt)
        dec = SynaspotDecoder(str(ckpt), device=args.device)
        dec.enroll(audio_path=rec0["wav"], text=rec0["text"])
        res = dec.detect(rec0["wav"], chunk_frames=20)
        s = res["score"]
        print(f"[6] 解码自测: 注册词='{rec0['text']}' 帧数={res['num_frames']} "
              f"score min={s.min():.4f} max={s.max():.4f} 末帧={s[-1]:.4f}")
        assert res["num_frames"] > 0 and torch.isfinite(s).all()

    print("\n冒烟验证全部通过 ✔")


if __name__ == "__main__":
    main()
