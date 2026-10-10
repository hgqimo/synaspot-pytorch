"""路B: 只蒸馏在线音频编码器 + 复用教师文本/混合分支 (独立脚本, 不依赖 train.py).

与 train.py 完全解耦: 不改动任何现有文件, 只 import 共享的模型/损失/数据组件
(model / losses / dataset / config). 训练循环与"复用教师分支"逻辑均在本文件内.

背景:
- 教师 = 离线强编码器(ssl HuBERT), 已训好, 冻结, 仅作监督源.
- 学生 = 在线因果编码器(dfsmn), 从零训练, 部署时轻量且可流式.
- 蒸馏信号只作用于音频编码器: E_A 逐帧余弦对齐 + 音素 Logit KD.
- 训完把教师的 text_encoder / mixed 直接搬进学生——l_demb 已把学生音频空间
  对齐到教师, 而教师文本/混合分支本就工作在该空间, 无需重训. 产出完整可部署检查点.

用法:
    python -m synaspot.distill_online \
        --teacher-ckpt checkpoints/ssl/mixed/best.pt \
        --save-dir checkpoints/dfsmn_online \
        --epochs 30 --batch-size 64

从零 dfsmn 学生易塌缩(acc_ph 卡在众数水平), 根因已定位: 训练中 e_a 输出有效秩
从 81 塌缩到 3~5(奇异值 610 vs 53), 24 类音素无从展开. 本脚本默认加 --w-rank 谱熵
秩保持正则直接对抗; 仍失败时降低 --aam-scale 8~12、放大 --enc-lr-mult 3~5.
--no-distill 只训音素+秩正则, 用于隔离验证.

产出检查点格式与 train.py 一致, 可直接用 decode.py / eval_wenetphrase.py 加载.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig
from .dataset import BucketBatchSampler, SynaspotDataset, collate_fn
from .losses import frame_align_loss, logit_kd_loss, phoneme_loss
from .model import Synaspot

# 默认数据目录: <项目根>/dataset/AISHELL-1/processed/synaspot (可用 --data-dir 覆盖)
DEFAULT_DATA = str(Path(__file__).resolve().parent.parent / "dataset"
                   / "AISHELL-1" / "processed" / "synaspot")


# ---------------------------------------------------------------- 教师载入
def load_teacher(teacher_ckpt: str, teacher_backbone: str, student_cfg: ModelConfig,
                 device: torch.device):
    """载入冻结的离线教师, 并校验与学生的一致性(帧率/维度/词表/主干)."""
    ck = torch.load(teacher_ckpt, map_location="cpu")
    tcfg = ModelConfig.from_dict(ck["config"])
    if tcfg.audio_backbone != teacher_backbone:
        raise ValueError(f"教师训练主干 {tcfg.audio_backbone} != "
                         f"--teacher-backbone {teacher_backbone}, 离线特征会对不上")
    if tcfg.feat_frame_rate != student_cfg.feat_frame_rate:
        raise ValueError(f"教师帧率 {tcfg.feat_frame_rate}Hz != 学生 "
                         f"{student_cfg.feat_frame_rate}Hz, E_A 无法逐帧对齐")
    if tcfg.hidden_dim != student_cfg.hidden_dim:
        raise ValueError(f"教师 hidden_dim {tcfg.hidden_dim} != 学生 "
                         f"{student_cfg.hidden_dim}, 嵌入级蒸馏维度不匹配")
    if tcfg.vocab_size != student_cfg.vocab_size:
        raise ValueError(f"教师词表 {tcfg.vocab_size} != 学生 {student_cfg.vocab_size}, "
                         f"Logit KD 需同一音素粒度")
    teacher = Synaspot(tcfg).to(device)
    # 跳过 log_priors / class_weight 注入缓冲, 避免空缓冲 vs 已注入缓冲的形状错配
    sd = {k: v for k, v in ck["model"].items()
          if "log_priors" not in k and "class_weight" not in k}
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"[教师] 载入 {teacher_ckpt}: backbone={tcfg.audio_backbone} {n:.2f}M "
          f"(missing={len(missing)}, unexpected={len(unexpected)}), 已冻结")
    return teacher, tcfg


# ---------------------------------------------------------------- 先验 (Logit Adjustment)
def get_log_priors(data_dir: Path, vocab_size: int, ignore=(0, 1, 2)):
    """统计训练集帧级标签先验取 log (带磁盘缓存), 供 Logit Adjustment 抗长尾塌缩."""
    cache = data_dir / "label_priors.pt"
    if cache.exists():
        t = torch.load(cache, map_location="cpu")
        if t.numel() == vocab_size:
            return t
    counts = np.zeros(vocab_size, dtype=np.int64)
    for f in sorted((data_dir / "labels" / "train").glob("*.npy")):
        counts += np.bincount(np.load(f), minlength=vocab_size)
    pc = counts.astype(np.float64).copy()
    for c in ignore:
        pc[c] = 0.0
    priors = pc / max(pc.sum(), 1.0)
    log_priors = np.log(np.clip(priors, 1e-12, None))
    for c in ignore:
        log_priors[c] = -100.0
    t = torch.from_numpy(log_priors).float()
    torch.save(t, cache)
    return t


# ---------------------------------------------------------------- 损失
def spectral_entropy_loss(e: torch.Tensor):
    """秩保持正则: 最大化 batch 内 e_a 的谱熵, 直接对抗实测的"编码器输出塌缩到近1维"
    (塌缩 dfsmn 的 e_a 奇异值 610 vs 53, 有效秩仅 3~5, 24 类音素无从展开 -> 众数塌缩;
    而 ssl 教师 e_a 有效秩 24~30, 随机初始化 dfsmn 为 81).

    e: (N, D) 帧级嵌入(通常已 LayerNorm). 返回 0=满秩(谱均匀), 越大=越塌缩.
    只作用于训练损失, 不改模型结构, 不影响教师权重加载.
    """
    e = e.float()
    e = e - e.mean(0)
    cov = (e.T @ e) / max(e.shape[0], 1)
    D = e.shape[1]
    sv = torch.linalg.svdvals(cov + 1e-6 * torch.eye(D, device=e.device))
    p = sv / sv.sum().clamp_min(1e-8)
    p = p.clamp_min(1e-6)
    ent = -(p * p.log()).sum() / math.log(max(D, 2))
    return 1.0 - ent.clamp(0.0, 1.0)


def audio_distill_loss(student: Synaspot, teacher: Synaspot, batch: dict,
                       cfg: ModelConfig, args):
    """audio 阶段的蒸馏损失: l_ph + w_rank*l_rank + w_embed*l_demb + w_logit*l_dlogit.

    l_ph   : 帧级音素 AAM 损失(地面真值, 内容帧).
    l_rank : 谱熵秩保持正则(全有效帧), 对抗 e_a 秩塌缩导致的众数塌缩.
    l_demb : 学生 E_A 逐帧余弦对齐教师 E_A(全有效帧).
    l_dlogit: 学生音素 logits 向教师软标签做 KD(内容帧, 干净 logits).
    --no-distill 时只保留 l_ph + l_rank, 用于隔离验证.
    """
    fbank, labels, mask = batch["fbank"], batch["labels"], batch["mask"]
    e_a = student.encode_audio(fbank)
    l_ph, acc_ph = phoneme_loss(student.phoneme_head, e_a, labels, mask)
    l_rank = spectral_entropy_loss(e_a[mask])
    logs = {"l_ph": l_ph.detach().item(), "acc_ph": acc_ph.detach().item(),
            "l_rank": l_rank.detach().item()}
    loss = l_ph + args.w_rank * l_rank

    if not args.no_distill:
        with torch.no_grad():
            t_ea = teacher.encode_audio(batch["teacher_fbank"])
        l_demb = frame_align_loss(e_a, t_ea.detach(), mask)

        content = mask & (labels >= 3)
        s_logits = student.phoneme_head(e_a, None)
        with torch.no_grad():
            t_logits = teacher.phoneme_head(t_ea, None)
        l_dlogit = logit_kd_loss(s_logits[content], t_logits[content],
                                 args.temperature)

        logs["l_demb"] = l_demb.detach().item()
        logs["l_dlogit"] = l_dlogit.detach().item()
        loss = loss + args.w_embed * l_demb + args.w_logit * l_dlogit
    return loss, logs


@torch.no_grad()
def evaluate(student: Synaspot, teacher: Synaspot, loader, cfg: ModelConfig, args):
    """dev 评估: 统一用目标间隔, 返回按帧数加权的指标均值."""
    student.eval()
    student.phoneme_head.set_margin(cfg.aam_margin)
    use_amp = args.precision == "bf16"
    tot, n = {}, 0
    for batch in loader:
        batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
        fbank, labels, mask = batch["fbank"], batch["labels"], batch["mask"]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            e_a = student.encode_audio(fbank)
            l_ph, acc_ph = phoneme_loss(student.phoneme_head, e_a, labels, mask)
            l_rank = spectral_entropy_loss(e_a[mask])
            t_ea = teacher.encode_audio(batch["teacher_fbank"])
            l_demb = frame_align_loss(e_a, t_ea, mask)
            content = mask & (labels >= 3)
            s_logits = student.phoneme_head(e_a, None)
            t_logits = teacher.phoneme_head(t_ea, None)
            l_dlogit = logit_kd_loss(s_logits[content], t_logits[content],
                                     args.temperature)
        nf = int(mask.sum().item())
        for k, v in (("l_ph", l_ph), ("acc_ph", acc_ph), ("l_rank", l_rank),
                     ("l_demb", l_demb), ("l_dlogit", l_dlogit)):
            tot[k] = tot.get(k, 0.0) + v.item() * nf
        n += nf
    return {k: v / max(n, 1) for k, v in tot.items()}


def schedule_factor(step: int, total_steps: int, warmup_steps: int):
    """线性 warmup + 余弦退火的调度因子(0~1), 各参数组乘各自基础 lr."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


# ---------------------------------------------------------------- 训练
def train(args, student: Synaspot, teacher: Synaspot, train_loader, dev_loader,
          cfg: ModelConfig):
    # 只训音频分支: 冻结文本/混合/说话人(蒸馏只针对音频编码器)
    for mod in (student.text_encoder, student.mixed, student.speaker_clf):
        for p in mod.parameters():
            p.requires_grad_(False)
    # 编码器与音素头分离学习率: 分类损失(∝aam_scale)易压垮编码器梯度, 编码器可单独放大 lr
    enc_params = [p for n, p in student.named_parameters()
                  if p.requires_grad and n.startswith("audio_encoder")]
    head_params = [p for n, p in student.named_parameters()
                   if p.requires_grad and not n.startswith("audio_encoder")]
    n_train = sum(p.numel() for p in enc_params + head_params) / 1e6
    print(f"[学生] 可训练参数量: {n_train:.2f}M (audio_encoder + phoneme_head)")

    groups = []
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr * args.enc_lr_mult})
    if head_params:
        groups.append({"params": head_params, "lr": args.lr})
    optimizer = torch.optim.Adam(groups)
    group_base = [g["lr"] for g in optimizer.param_groups]
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    use_amp = args.precision == "bf16"

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(args.data_dir) / "pinyin_vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    best_acc = -1.0
    best_state = None
    no_improve = 0
    gstep = 0
    for epoch in range(1, args.epochs + 1):
        student.train()
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        t0 = time.time()
        run, nb = {}, 0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
            progress = (epoch - 1 + (step - 1) / max(1, len(train_loader))) / args.epochs
            margin = cfg.aam_margin * min(1.0, progress / max(1e-6, args.margin_ramp))
            student.phoneme_head.set_margin(margin)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, logs = audio_distill_loss(student, teacher, batch, cfg, args)
            logs["margin"] = margin
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                print(f"[audio] ep{epoch} step{step}: 非有限损失, 跳过本步")
                continue
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            logs["gnorm"] = float(gnorm) if torch.isfinite(gnorm) else 999.0
            if not torch.isfinite(gnorm):
                optimizer.zero_grad(set_to_none=True)
                print(f"[audio] ep{epoch} step{step}: 梯度非有限, 跳过更新")
                continue
            factor = schedule_factor(gstep, total_steps, args.warmup_steps)
            for g, base in zip(optimizer.param_groups, group_base):
                g["lr"] = base * factor
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            gstep += 1

            for k, v in logs.items():
                run[k] = run.get(k, 0.0) + v
            nb += 1
            if step % args.log_interval == 0:
                msg = " ".join(f"{k}={v / nb:.4f}" for k, v in run.items())
                print(f"[audio] ep{epoch} {step}/{len(train_loader)} {msg} lr={lr:.2e}")
                run, nb = {}, 0

        # dev 评估 + 早停
        dev = evaluate(student, teacher, dev_loader, cfg, args)
        msg = " ".join(f"{k}={v:.4f}" for k, v in dev.items())
        print(f"[audio] ep{epoch} dev: {msg} ({time.time() - t0:.0f}s)")
        watch = dev["acc_ph"]
        if watch > best_acc:
            best_acc = watch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in student.state_dict().items()}
            no_improve = 0
            torch.save({"model": best_state, "config": cfg.to_dict(), "vocab": vocab},
                       save_dir / "audio_best.pt")
            print(f"[audio] 新最优, 已保存 audio_best.pt (acc_ph={best_acc:.4f})")
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f"[audio] 连续 {args.patience} 轮 acc_ph 无提升, 早停于 ep{epoch} "
                  f"(best acc_ph={best_acc:.4f})")
            break

    if best_state is None:
        raise RuntimeError("训练未产生任何有效检查点")
    return best_state


# ---------------------------------------------------------------- 复用教师分支
def build_final(student_cfg: ModelConfig, best_state: dict, teacher: Synaspot,
                vocab: dict, save_dir: Path):
    """学生(蒸馏后的音频编码器) + 教师(text_encoder / mixed) -> 完整可部署模型."""
    final = Synaspot(student_cfg)
    sd = {k: v for k, v in best_state.items()
          if "log_priors" not in k and "class_weight" not in k}
    final.load_state_dict(sd, strict=False)
    # 复用教师文本/混合分支: 它们对齐教师音频空间, 而蒸馏已把学生音频空间拉近教师.
    final.text_encoder.load_state_dict(teacher.text_encoder.state_dict())
    final.mixed.load_state_dict(teacher.mixed.state_dict())

    out = save_dir / "student_online.pt"
    torch.save({"model": final.state_dict(), "config": student_cfg.to_dict(),
                "vocab": vocab}, out)
    n = sum(p.numel() for p in final.parameters()) / 1e6
    print(f"[完成] 可部署在线学生已保存: {out}")
    print(f"        audio_encoder=学生(dfsmn, 蒸馏), text_encoder/mixed=教师复用, "
          f"总参数量 {n:.2f}M")
    print(f"        评估: python -m synaspot.eval_wenetphrase --ckpt {out} "
          f"--modes T,A,TA --max-keywords 300")
    return out


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(
        description="路B: 只蒸馏在线音频编码器 + 复用教师文本/混合分支")
    ap.add_argument("--teacher-ckpt", required=True, help="教师检查点(离线 ssl 训好)")
    ap.add_argument("--teacher-backbone", choices=["ssl"], default="ssl",
                    help="教师主干=离线特征来源, 须与教师训练时一致")
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--save-dir", default=str(Path(DEFAULT_DATA).parent / "exp_online"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--enc-lr-mult", type=float, default=1.0,
                    help="编码器参数组 lr 倍数(音素头为 1x), 放大可加速编码器收敛")
    ap.add_argument("--aam-scale", type=float, default=None,
                    help="覆盖学生 AAM 缩放(默认继承教师), 塌缩时可降到 8~12")
    ap.add_argument("--aam-margin", type=float, default=None,
                    help="覆盖学生 AAM margin(默认继承教师), 从零训练建议 0")
    ap.add_argument("--proj-dim", type=int, default=None,
                    help="覆盖 DFSMN 层内瓶颈维度(默认 128). hidden_dim 固定 256 以对齐教师, "
                         "扩容走 proj_dim/num_layers: proj_dim=256 时约 1.0M 参数")
    ap.add_argument("--num-layers", type=int, default=None,
                    help="覆盖 DFSMN 层数(默认 7), 与 --proj-dim 配合扩容")
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--w-embed", type=float, default=0.5, help="E_A 嵌入级蒸馏权重")
    ap.add_argument("--w-logit", type=float, default=0.1, help="Logit KD 权重")
    ap.add_argument("--w-rank", type=float, default=1.0,
                    help="谱熵秩保持正则权重(对抗实测的 e_a 秩塌缩->众数塌缩, 0=关)")
    ap.add_argument("--no-distill", action="store_true",
                    help="只训音素+秩正则(不加载蒸馏信号), 用于隔离验证塌缩修复")
    ap.add_argument("--temperature", type=float, default=4.0, help="KD 温度")
    ap.add_argument("--margin-ramp", type=float, default=0.3,
                    help="AAM margin 从 0 升到目标值的进度占比")
    ap.add_argument("--patience", type=int, default=8, help="平台早停耐心")
    ap.add_argument("--max-frames", type=int, default=1600)
    ap.add_argument("--dither", type=float, default=0.0,
                    help="在线 fbank 抖动强度(kaldi dither). 本库 log-mel 特征值域小, "
                         "默认 kaldi dither=1.0 会引入与信号同量级的噪声淹没特征(实测 "
                         "diff std 3.3 vs 信号 std 3.2), 从零训练应设 0; 有预训练可试 <0.1")
    ap.add_argument("--use-delta", action="store_true",
                    help="fbank 后拼接 Δ/ΔΔ (80->240 维), 补声母瞬态判别信息, 突破 "
                         "静态 fbank 相对 HuBERT 的特征信息量瓶颈")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--la-tau", type=float, default=0.0, help="Logit Adjustment 温度(抗长尾)")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--train-limit", type=int, default=0,
                    help=">0 时只用前 N 条训练(过拟合自检)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # 学生配置: 继承教师(帧率/维度/词表/粒度一致), 只换主干为因果 dfsmn
    ck = torch.load(args.teacher_ckpt, map_location="cpu")
    teacher_cfg = ModelConfig.from_dict(ck["config"])
    student_cfg = ModelConfig.from_dict(teacher_cfg.to_dict())
    student_cfg.audio_backbone = "dfsmn"
    student_cfg.distill_enabled = False
    student_cfg.distill_teacher_ckpt = ""
    student_cfg.use_delta = args.use_delta
    if args.aam_scale is not None:
        student_cfg.aam_scale = args.aam_scale
    if args.aam_margin is not None:
        student_cfg.aam_margin = args.aam_margin
    if args.proj_dim is not None:
        student_cfg.proj_dim = args.proj_dim
    if args.num_layers is not None:
        student_cfg.num_layers = args.num_layers
    print(f"[学生] backbone={student_cfg.audio_backbone} "
          f"feat_frame_rate={student_cfg.feat_frame_rate}Hz "
          f"hidden={student_cfg.hidden_dim} vocab={student_cfg.vocab_size} "
          f"granularity={student_cfg.phoneme_granularity}")

    teacher, tcfg = load_teacher(args.teacher_ckpt, args.teacher_backbone,
                                 student_cfg, device)
    assert tcfg.feat_frame_rate == student_cfg.feat_frame_rate

    student = Synaspot(student_cfg).to(device)
    # Logit Adjustment 抗长尾塌缩 (注入须在模型构建之后, 避免被后续 load 覆盖)
    if args.la_tau > 0:
        log_priors = get_log_priors(Path(args.data_dir), student_cfg.vocab_size)
        student.phoneme_head.set_log_priors(log_priors.to(device), tau=args.la_tau,
                                            label_smoothing=args.label_smoothing)
        print(f"[学生] Logit Adjustment tau={args.la_tau}, "
              f"label_smoothing={args.label_smoothing}")

    # 数据: 学生 dfsmn 在线 fbank(feat_dir=None), 教师读离线缓存特征
    tsub = "hubert_feats"
    if not (Path(args.data_dir) / tsub / "train").exists():
        raise FileNotFoundError(f"未找到 {Path(args.data_dir) / tsub / 'train'}: "
                                f"教师主干 {args.teacher_backbone} 需先离线提特征")

    def make_loader(split, training):
        ds = SynaspotDataset(
            str(Path(args.data_dir) / f"{split}.jsonl"), student_cfg,
            max_frames=args.max_frames, training=training,
            limit=args.train_limit if (training and args.train_limit > 0) else 0,
            dither=args.dither,
            feat_dir=None,
            teacher_feat_dir=str(Path(args.data_dir) / tsub / split),
            teacher_backbone=args.teacher_backbone)
        if training:
            overfit = args.train_limit > 0
            sampler = BucketBatchSampler(ds.sizes, args.batch_size,
                                         shuffle=not overfit, drop_last=not overfit,
                                         rank=0, world_size=1, seed=args.seed)
            return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn,
                              num_workers=args.num_workers, pin_memory=True)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=args.num_workers,
                          pin_memory=True)

    train_loader = make_loader("train", training=True)
    dev_loader = make_loader("dev", training=False)
    print(f"[数据] train={len(train_loader)} 批, dev={len(dev_loader)} 批")

    best_state = train(args, student, teacher, train_loader, dev_loader, student_cfg)

    with open(Path(args.data_dir) / "pinyin_vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    build_final(student_cfg, best_state, teacher, vocab, Path(args.save_dir))


if __name__ == "__main__":
    main()
