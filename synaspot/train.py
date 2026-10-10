"""SYNASPOT 两阶段训练.

阶段一 (audio):  L_audio = α_A·L_ph + β_A·L_vp          (论文 Function 2)
阶段二 (mixed):  L_mixed = α_M·L_ph + β_M·L_clat + γ_M·L_clam (论文 Function 5)
                 从阶段一最优权重初始化, 联合训练文本/混合分支.

论文 3.2: 50 epochs, Adam lr=1e-3, annealing factor 0.3, batch size 100.

单机单卡:  python train.py --stage both
单机多卡:  torchrun --nproc_per_node=4 train.py --stage both
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import ModelConfig
from .dataset import BucketBatchSampler, SynaspotDataset, collate_fn
from .losses import contrastive_loss, frame_align_loss, logit_kd_loss, phoneme_loss
from .model import Synaspot

# 默认数据目录: <项目根>/dataset/AISHELL-1/processed/synaspot (可用 --data-dir 覆盖)
DEFAULT_DATA = str(Path(__file__).resolve().parent.parent / "dataset"
                   / "AISHELL-1" / "processed" / "synaspot")
# 默认检查点目录: <项目根>/checkpoints (可用 --save-dir 覆盖).
# 检查点与数据集分离存放: 训练产物不再混入 dataset/, 便于清理与备份.
DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "checkpoints"


def get_log_priors(data_dir: Path, vocab_size: int, ignore=(0, 1, 2)):
    """统计训练集帧级标签先验并取 log, 供 Logit Adjustment 用. 带磁盘缓存.

    只在缓存不存在时扫一遍 labels/train (约12万个 npy), 算完落盘;
    之后所有进程/阶段直接读缓存, 避免 DDP 竞态与重复统计.
    """
    import numpy as np
    cache = Path(data_dir) / "label_priors.pt"
    if cache.exists():
        t = torch.load(cache, map_location="cpu")
        if t.numel() == vocab_size:
            return t
        # 词表规模变了(换了音素粒度), 旧缓存失效, 落到下方重算
    counts = np.zeros(vocab_size, dtype=np.int64)
    files = sorted((Path(data_dir) / "labels" / "train").glob("*.npy"))
    for f in files:
        counts += np.bincount(np.load(f), minlength=vocab_size)
    pc = counts.astype(np.float64).copy()
    for c in ignore:
        pc[c] = 0.0
    priors = pc / max(pc.sum(), 1.0)
    log_priors = np.log(np.clip(priors, 1e-12, None))
    for c in ignore:
        log_priors[c] = -100.0  # 特殊类永不作为预测目标
    t = torch.from_numpy(log_priors).float()
    torch.save(t, cache)
    return t


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world, True
    return 0, 1, False


def load_teacher(cfg: ModelConfig, device, rank: int):
    """载入冻结的离线教师 (用 distill_teacher_backbone 主干预先训好的完整模型).

    蒸馏前提校验, 任一不符即报错(避免静默错配):
    - 帧率一致 -> E_A 逐帧对齐;  hidden_dim 一致 -> 嵌入级余弦维度匹配;
    - 词表一致 -> Logit KD 同一音素粒度;  教师训练主干 == distill_teacher_backbone.
    """
    ck = torch.load(cfg.distill_teacher_ckpt, map_location="cpu")
    tcfg = ModelConfig.from_dict(ck["config"])
    if tcfg.feat_frame_rate != cfg.feat_frame_rate:
        raise ValueError(f"教师帧率 {tcfg.feat_frame_rate}Hz != 学生 "
                         f"{cfg.feat_frame_rate}Hz, E_A 无法逐帧对齐")
    if tcfg.hidden_dim != cfg.hidden_dim:
        raise ValueError(f"教师 hidden_dim {tcfg.hidden_dim} != 学生 "
                         f"{cfg.hidden_dim}, E_A 嵌入级蒸馏维度不匹配")
    if tcfg.vocab_size != cfg.vocab_size:
        raise ValueError(f"教师词表 {tcfg.vocab_size} != 学生 {cfg.vocab_size}, "
                         f"Logit KD 需同一音素粒度")
    if tcfg.audio_backbone != cfg.distill_teacher_backbone:
        raise ValueError(f"教师训练主干 {tcfg.audio_backbone} != "
                         f"distill_teacher_backbone {cfg.distill_teacher_backbone}, "
                         f"离线特征会对不上")
    teacher = Synaspot(tcfg).to(device)
    missing, unexpected = teacher.load_state_dict(ck["model"], strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if rank == 0:
        n = sum(p.numel() for p in teacher.parameters()) / 1e6
        print(f"[distill] 载入教师: backbone={tcfg.audio_backbone} {n:.2f}M "
              f"(missing={len(missing)}, unexpected={len(unexpected)}), 已冻结")
    return teacher


def validate_config(cfg: ModelConfig, info: dict, data_dir: Path, rank: int):
    """训练之初的一致性校验: 帧率×主干可整除重采样, 粒度与数据集词表自洽."""
    native, rate = cfg.backbone_native_hz, cfg.feat_frame_rate
    if native > rate:
        if native % rate != 0:
            raise ValueError(
                f"主干 {cfg.audio_backbone}({native}Hz) 无法整数降采样到 {rate}Hz")
    elif native < rate:
        if rate % native != 0:
            raise ValueError(
                f"主干 {cfg.audio_backbone}({native}Hz) 无法整数升采样到 {rate}Hz")
    # 粒度与数据集一致性 (dataset_info.json 有记录时)
    data_gran = info.get("phoneme_granularity")
    if data_gran and data_gran != cfg.phoneme_granularity:
        raise ValueError(
            f"配置音素粒度 '{cfg.phoneme_granularity}' 与数据集 '{data_gran}' 不一致, "
            f"词表不匹配. 请重跑 04_build_dataset.py --granularity "
            f"{cfg.phoneme_granularity} 重建数据集, 或用 --phoneme-granularity {data_gran}")
    # ssl 需离线特征已提取
    if cfg.audio_backbone == "ssl":
        sub = "hubert_feats"
        if not (data_dir / sub / "train").exists():
            raise FileNotFoundError(
                f"未找到 {data_dir / sub / 'train'}: ssl 主干需先跑 "
                f"extract_ssl.py 离线提特征")
    # 蒸馏前置: 教师检查点存在 + 教师主干离线特征已提取 (帧率/维度/词表在 load_teacher 校验)
    if cfg.distill_enabled:
        if not Path(cfg.distill_teacher_ckpt).exists():
            raise FileNotFoundError(
                f"未找到教师检查点 {cfg.distill_teacher_ckpt}: 请先用 "
                f"--audio-backbone {cfg.distill_teacher_backbone} 训好教师模型")
        tsub = "hubert_feats"
        if not (data_dir / tsub / "train").exists():
            raise FileNotFoundError(
                f"未找到 {data_dir / tsub / 'train'}: 蒸馏教师主干 "
                f"{cfg.distill_teacher_backbone} 需先离线提特征")
    if rank == 0:
        distill_msg = ""
        if cfg.distill_enabled:
            distill_msg = (f" | 蒸馏: 教师={cfg.distill_teacher_backbone}"
                           f"(w_e={cfg.distill_w_embed}, w_l={cfg.distill_w_logit}, "
                           f"T={cfg.distill_temperature})")
        print(f"[配置] backbone={cfg.audio_backbone} feat_frame_rate={rate}Hz "
              f"(原生 {native}Hz, 标签降采样 x{cfg.label_ds}) "
              f"granularity={cfg.phoneme_granularity} vocab={cfg.vocab_size}"
              f"{distill_msg}")


def make_loader(data_dir: Path, cfg: ModelConfig, split: str, args,
                rank: int, world: int):
    # 过拟合自检: train_limit>0 时只取前 N 条, 关 dither/shuffle/drop_last,
    # 让模型反复看同一小撮样本——拟合得动=架构/标签没问题, 是优化问题;
    # 拟合不动=架构或帧级标签有 bug.
    overfit = (split == "train") and getattr(args, "train_limit", 0) > 0
    # ssl 主干: 指向离线缓存特征目录 (extract_ssl.py)
    feat_dir = None
    if cfg.audio_backbone == "ssl":
        feat_dir = str(data_dir / "hubert_feats" / split)
    # 蒸馏: 教师主干(离线)特征目录. 教师主干须为有离线缓存的 ssl.
    teacher_feat_dir, teacher_backbone = None, None
    if cfg.distill_enabled:
        teacher_backbone = cfg.distill_teacher_backbone
        if teacher_backbone != "ssl":
            raise ValueError(f"蒸馏教师主干须为 ssl(离线缓存), "
                             f"got {teacher_backbone}")
        tsub = "hubert_feats"
        teacher_feat_dir = str(data_dir / tsub / split)
    ds = SynaspotDataset(str(data_dir / f"{split}.jsonl"), cfg,
                         max_frames=args.max_frames, training=(split == "train"),
                         limit=args.train_limit if split == "train" else 0,
                         dither=getattr(args, "dither", 0.0), feat_dir=feat_dir,
                         teacher_feat_dir=teacher_feat_dir,
                         teacher_backbone=teacher_backbone)
    if split == "train":
        sampler = BucketBatchSampler(ds.sizes, args.batch_size,
                                     shuffle=not overfit,
                                     drop_last=not overfit,
                                     rank=rank, world_size=world,
                                     seed=args.seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn,
                          num_workers=args.num_workers, pin_memory=True), sampler
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.num_workers,
                        pin_memory=True)
    return loader, None


def _distill_losses(model, teacher, batch, cfg, e_a, labels, mask):
    """学生对齐冻结教师: E_A 嵌入级(逐帧余弦, 全部有效帧) + Logit 级 KD(内容帧).

    教师前向在 no_grad 下不产生梯度; 学生 e_a/phoneme_head 可训. 返回 (l_demb, l_dlogit).
    """
    m = unwrap(model)
    with torch.no_grad():
        t_ea = teacher.encode_audio(batch["teacher_fbank"])
    # 嵌入级: 逐帧余弦对齐. 教师音频编码器全帧都有意义(含静音), 故用全部有效帧.
    l_demb = frame_align_loss(e_a, t_ea.detach(), mask)
    # Logit 级 KD: 只在内容帧(排除 <PAD>/<UNK>/<SIL>)——音素头特殊类权重未被
    # 音素损失训练过, 其教师 logits 是噪声. 软标签用干净 s*cos(targets=None, 无 margin/LA).
    content = mask & (labels >= 3)
    s_logits = m.phoneme_head(e_a, None)
    with torch.no_grad():
        t_logits = teacher.phoneme_head(t_ea, None)
    l_dlogit = logit_kd_loss(s_logits[content], t_logits[content],
                             cfg.distill_temperature)
    return l_demb, l_dlogit


def stage_loss(model: Synaspot, batch, cfg: ModelConfig, stage: str,
               grl_alpha: float = 1.0, eval_mode: bool = False,
               teacher: Synaspot = None):
    """返回 (loss, log_dict). eval_mode=True 时跳过说话人损失
    (dev/test 说话人不在训练说话人表中, spk_id=-1).
    teacher 非 None 且非 eval 时, 在 audio/mixed 阶段叠加离线→在线蒸馏损失."""
    fbank = batch["fbank"]
    labels = batch["labels"]
    mask = batch["mask"]
    spk = batch["spk_id"]

    # 所有计算必须经 forward 触发, 否则 DDP 梯度同步失效
    outs = model(fbank, labels, mask, stage, grl_alpha)
    e_a = outs["e_a"]

    # text-only 对齐阶段: 音频编码器冻结, 只把 e_t 拉向 e_a(解 InfoNCE 冷启动).
    # 主损失=逐帧余弦对齐; 可选叠加共享音素分类器蒸馏(align_cls_w>0).
    if stage == "text":
        e_t = outs["e_t"]
        content = mask & (labels >= 3)
        l_align = frame_align_loss(e_t, e_a.detach(), content)
        logs = {"l_align": l_align.detach().item()}
        with torch.no_grad():
            src_n = F.normalize(e_t[content].float(), dim=-1)
            tgt_n = F.normalize(e_a[content].float(), dim=-1)
            logs["cos_ta"] = (src_n * tgt_n).sum(-1).mean().item()
        loss = l_align
        if cfg.align_cls_w > 0:
            l_cls, acc_cls = phoneme_loss(unwrap(model).phoneme_head, e_t,
                                          labels, mask)
            loss = loss + cfg.align_cls_w * l_cls
            logs["l_tcls"] = l_cls.detach().item()
            logs["acc_tcls"] = acc_cls.detach().item()
        return loss, logs

    l_ph, acc_ph = phoneme_loss(unwrap(model).phoneme_head, e_a, labels, mask)
    logs = {"l_ph": l_ph.detach().item(), "acc_ph": acc_ph.detach().item()}

    # 离线→在线蒸馏: 学生 e_a/logits 对齐冻结教师 (仅训练, audio/mixed 阶段)
    l_distill = None
    if teacher is not None and not eval_mode:
        l_demb, l_dlogit = _distill_losses(model, teacher, batch, cfg, e_a,
                                           labels, mask)
        logs["l_demb"] = l_demb.detach().item()
        logs["l_dlogit"] = l_dlogit.detach().item()
        l_distill = (cfg.distill_w_embed * l_demb
                     + cfg.distill_w_logit * l_dlogit)

    if stage == "audio":
        if eval_mode:
            return l_ph, logs
        logits = outs["spk_logits"]
        l_vp = F.cross_entropy(logits, spk)
        logs["l_vp"] = l_vp.detach().item()
        logs["acc_spk"] = (logits.argmax(-1) == spk).float().mean().item()
        loss = cfg.alpha_A * l_ph + cfg.beta_A * l_vp
        if l_distill is not None:
            loss = loss + l_distill
        return loss, logs

    # stage == "mixed"
    e_t, e_m = outs["e_t"], outs["e_m"]
    # 排除 <PAD>/<UNK>/<SIL> 帧对齐静音太容易, 稀释内容帧信号.
    content = mask & (labels >= 3)
    # 论文 Function 3/4 纯 InfoNCE: 不做 same-label 屏蔽(所有 j 都是负样本),
    # same-label mask 会去掉同类负样本, 弱化判别信号.
    l_clat = contrastive_loss(e_t, e_a, content, cfg.temperature, None,
                              cfg.cl_max_frames)
    l_clam = contrastive_loss(e_m, e_a, content, cfg.temperature, None,
                              cfg.cl_max_frames)
    logs["l_clat"] = l_clat.detach().item()
    logs["l_clam"] = l_clam.detach().item()
    loss = cfg.alpha_M * l_ph + cfg.beta_M * l_clat + cfg.gamma_M * l_clam
    if l_distill is not None:
        loss = loss + l_distill
    return loss, logs


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def freeze_unused_branches(model, stage: str):
    """冻结当前阶段不参与的分支, 避免 DDP 等待未使用参数的梯度.

    audio 阶段: 文本/混合分支不参与; text 阶段: 只训文本编码器;
    mixed 阶段: 说话人分类器不参与. 需在 DDP 包装前调用.
    """
    m = unwrap(model)
    for p in m.parameters():
        p.requires_grad_(True)
    if stage == "audio":
        for mod in (m.text_encoder, m.mixed):
            for p in mod.parameters():
                p.requires_grad_(False)
    elif stage == "text":
        # 只训 text_encoder: 冻结音频编码器/音素头/说话人/混合分支,
        # 音频嵌入作为固定对齐目标(见 stage_loss 的 e_a.detach()).
        for mod in (m.audio_encoder, m.phoneme_head, m.speaker_clf, m.mixed):
            for p in mod.parameters():
                p.requires_grad_(False)
    else:
        for p in m.speaker_clf.parameters():
            p.requires_grad_(False)
        # 不冻结音频编码器: 论文 2.1.2 "jointly optimize", 音频编码器在阶段3
        # 联合微调, 靠 L_ph(权重0.4)保住音素特征. 冻结会让文本编码器梯度
        # 极小(~2e-4)而 l_clat 卡死在 log(N) 随机水平.


def grl_schedule(progress: float, start: float, ramp: float) -> float:
    """对抗梯度反转强度: 先为0(纯学音素), 再线性升到1.

    progress: 训练进度 [0,1]; 返回 grl_alpha ∈ [0,1].
    """
    if progress < start:
        return 0.0
    return min(1.0, (progress - start) / max(1e-6, ramp))


def param_grad_norm(params):
    """参数组当前 .grad 的 L2 范数(裁剪前)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().float().norm().item() ** 2
    return total ** 0.5


def grad_conflict_probe(model, batch, cfg, grl_alpha, use_amp):
    """分解 音素/声纹 损失在音频编码器上的梯度范数与夹角余弦(诊断对抗冲突).

    用独立前向 + autograd.grad, 不累积到 .grad, 不触发 DDP 梯度同步.
    返回 {g_ph_enc, g_vp_enc, cos_ph_vp}: cos 为负=方向冲突, g_vp>>g_ph=对抗压制.
    """
    m = unwrap(model)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        outs = m(batch["fbank"], batch["labels"], batch["mask"], "audio", grl_alpha)
        e_a = outs["e_a"]
        l_ph, _ = phoneme_loss(m.phoneme_head, e_a, batch["labels"], batch["mask"])
        l_vp = F.cross_entropy(outs["spk_logits"], batch["spk_id"])
    enc = [p for p in m.audio_encoder.parameters() if p.requires_grad]
    g_ph = torch.autograd.grad(cfg.alpha_A * l_ph, enc, retain_graph=True,
                               allow_unused=True)
    g_vp = torch.autograd.grad(cfg.beta_A * l_vp, enc, retain_graph=False,
                               allow_unused=True)

    def cat(gs):
        return torch.cat([g.reshape(-1).float() for g in gs if g is not None])
    vph, vvp = cat(g_ph), cat(g_vp)
    n_ph, n_vp = vph.norm().item(), vvp.norm().item()
    cos = F.cosine_similarity(vph[None], vvp[None]).item()
    return {"g_ph_enc": n_ph, "g_vp_enc": n_vp, "cos_ph_vp": cos}


def build_cosine_scheduler(optimizer, args, total_steps: int):
    """warmup + 余弦退火 (按优化器步). 避免 ReduceLROnPlateau 过早塌缩."""
    import math
    warmup = max(1, args.warmup_steps)
    min_ratio = max(1e-6, args.min_lr / args.lr)

    def lam(step: int) -> float:
        step = max(1, step)
        if step < warmup:
            return step / warmup
        prog = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)


def build_param_groups(model, args):
    """编码器与其余分支分离学习率.

    编码器表征同时被音素/声纹/对比多个目标拉扯, 单一 lr 下梯度被摊薄、容易"饿死";
    给编码器 enc_lr_mult 倍的 lr 让它先学到帧级判别表征. mult=1.0 时退化为单一学习率.
    """
    m = unwrap(model)
    mult = getattr(args, "enc_lr_mult", 1.0)
    trainable = [p for p in m.parameters() if p.requires_grad]
    if mult == 1.0:
        return [{"params": trainable}]
    enc_params = [p for p in m.audio_encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in enc_params}
    other_params = [p for p in trainable if id(p) not in enc_ids]
    groups = []
    if enc_params:  # 音频编码器未冻结时才有这组(阶段3 冻结后为空, 跳过)
        groups.append({"params": enc_params, "lr": args.lr * mult})
    if other_params:
        groups.append({"params": other_params, "lr": args.lr})
    return groups


def reduce_mean(v: float, world: int) -> float:
    if world == 1:
        return v
    t = torch.tensor([v], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t.item() / world)


@torch.no_grad()
def evaluate(model, loader, cfg, stage, device, use_amp: bool = True):
    model.eval()
    tot, n = {}, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, logs = stage_loss(model, batch, cfg, stage, eval_mode=True)
        for k, v in logs.items():
            tot[k] = tot.get(k, 0.0) + v * labels_num(batch)
        tot["loss"] = tot.get("loss", 0.0) + loss.item() * labels_num(batch)
        n += labels_num(batch)
    return {k: v / max(n, 1) for k, v in tot.items()}


def labels_num(batch):
    return int(batch["mask"].sum().item())


def save_ckpt(path: Path, model, optimizer, scheduler, epoch: int, cfg, meta):
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        "model": state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "config": cfg.to_dict(),
        **meta,
    }, path)


def train_stage(stage: str, args, cfg, rank: int, world: int, ddp: bool,
                device, init_ckpt: Path = None, name: str = None,
                teacher: Synaspot = None):
    save_dir = Path(args.save_dir) / (name or stage)
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    writer = None
    if rank == 0 and not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=save_dir / "tb")
        writer.add_text("config", json.dumps(cfg.to_dict(), ensure_ascii=False,
                                             indent=2))

    with open(data_dir / "pinyin_vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    model = Synaspot(cfg).to(device)
    meta = {"vocab": vocab}
    if init_ckpt is not None and init_ckpt.exists():
        ck = torch.load(init_ckpt, map_location="cpu")
        # log_priors 是注入缓冲(空=未启用, 420=已启用), 跨阶段载入时形状可能不一致,
        # 直接整条跳过它, 随后由下方 set_log_priors 重新注入.
        sd = {k: v for k, v in ck["model"].items()
              if "log_priors" not in k and "class_weight" not in k}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if rank == 0:
            print(f"[{stage}] 从 {init_ckpt} 载入权重 "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
    # Logit Adjustment 抗多数类塌缩: 注入帧级标签先验(须在 load_state_dict 后,
    # 避免被检查点覆盖; tau=0 时不启用, 行为与原版完全一致)
    if getattr(args, "la_tau", 0.0) > 0:
        log_priors = get_log_priors(data_dir, cfg.vocab_size)
        model.phoneme_head.set_log_priors(
            log_priors.to(device), tau=args.la_tau,
            label_smoothing=getattr(args, "label_smoothing", 0.0))
        if rank == 0:
            print(f"[{stage}] Logit Adjustment tau={args.la_tau}, "
                  f"label_smoothing={getattr(args, 'label_smoothing', 0.0)}")
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{stage}] 模型参数量: {n_params / 1e6:.2f}M")

    train_loader, train_sampler = make_loader(data_dir, cfg, "train", args, rank, world)
    dev_loader, _ = make_loader(data_dir, cfg, "dev", args, rank, world)

    freeze_unused_branches(model, stage)
    if stage == "mixed" and getattr(args, "freeze_audio_mixed", False):
        m = unwrap(model)
        for mod in (m.audio_encoder, m.phoneme_head, m.speaker_clf):
            for p in mod.parameters():
                p.requires_grad_(False)
        if rank == 0:
            print(f"[{stage}] freeze_audio_mixed: 冻结 audio_encoder/phoneme_head/"
                  f"speaker_clf, 只训 text_encoder + mixed")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"[{stage}] 可训练参数量: {n_trainable / 1e6:.2f}M")
        if stage == "mixed" and not getattr(args, "freeze_audio_mixed", False):
            print(f"[{stage}] 联合优化(论文 jointly optimize): "
                  f"audio_encoder + text_encoder + mixed + phoneme_head")

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()])

    optimizer = torch.optim.Adam(build_param_groups(model, args))
    if rank == 0 and getattr(args, "enc_lr_mult", 1.0) != 1.0:
        print(f"[{stage}] 编码器 lr 分离: enc={args.lr * args.enc_lr_mult:.2e}, "
              f"heads={args.lr:.2e} (enc_lr_mult={args.enc_lr_mult})")
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = max(1, args.epochs * steps_per_epoch)
    if args.scheduler == "cosine":
        scheduler = build_cosine_scheduler(optimizer, args, total_steps)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=args.patience)

    # 平台早停监控指标: audio 看音素准确率(越高越好), text 看对齐损失(越低越好),
    # mixed 看总损失(越低越好)
    if stage == "audio":
        watch, mode = "acc_ph", "max"
    elif stage == "text":
        watch, mode = "l_align", "min"
    else:
        watch, mode = "loss", "min"
    start_epoch = 1
    best_val = float("-inf") if mode == "max" else float("inf")
    no_improve = 0
    resume = save_dir / "last.pt"
    if args.resume and resume.exists():
        ck = torch.load(resume, map_location="cpu")
        (model.module if ddp else model).load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", best_val)
        if rank == 0:
            print(f"[{stage}] 断点续训 epoch {start_epoch}, best {watch}={best_val:.4f}")

    use_amp = args.precision == "bf16"
    # 过拟合自检模式: 只用前 N 条训练样本反复拟合, 跳过昂贵 dev 评估、改盯 train 指标.
    # 用于判别"架构/帧级标签有 bug"(拟合不动) vs "优化/超参问题"(能拟合).
    overfit = getattr(args, "train_limit", 0) > 0
    # --no-dev-eval: 全量训练也跳过昂贵 dev(诊断/提速), 同样改盯 train 指标
    skip_dev = overfit or getattr(args, "no_dev_eval", False)
    gstep = 0  # 累计优化器步数, 作为 TensorBoard 横轴
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.time()
        run, win = {}, {}
        nb = nw = 0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            progress = ((epoch - 1) + (step - 1) / max(1, len(train_loader))) \
                / max(1, args.epochs)
            grl_alpha = grl_schedule(progress, args.grl_start, args.grl_ramp)
            grl_alpha = min(grl_alpha, args.grl_cap)
            # AAM margin warmup: audio 阶段从0渐进升到目标(先学音素再加间隔),
            # mixed 阶段沿用目标间隔
            sched_m = cfg.aam_margin * min(1.0, progress / max(1e-6, args.margin_ramp)) \
                if stage == "audio" else cfg.aam_margin
            unwrap(model).phoneme_head.set_margin(sched_m)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, logs = stage_loss(model, batch, cfg, stage,
                                        grl_alpha=grl_alpha, teacher=teacher)
                loss = loss / args.grad_accum
            logs["margin"] = sched_m
            # NaN 保护: 任一卡出现非有限损失时, 所有卡同步跳过本步更新
            bad = not torch.isfinite(loss).item()
            if world > 1:
                flag = torch.tensor([1.0 if bad else 0.0], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                bad = flag.item() > 0.0
            if bad:
                optimizer.zero_grad(set_to_none=True)
                if rank == 0:
                    print(f"[{stage}] ep{epoch} step{step}: 非有限损失, 跳过本步更新")
                continue
            loss.backward()
            if step % args.grad_accum == 0:
                # 裁剪前记录各分支梯度范数(诊断对抗冲突)
                mref = unwrap(model)
                if stage == "audio":
                    logs["g_enc"] = param_grad_norm(mref.audio_encoder.parameters())
                    logs["g_spk"] = param_grad_norm(mref.speaker_clf.parameters())
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                logs["gnorm"] = float(gnorm) if torch.isfinite(gnorm) else 999.0
                # 梯度 NaN 保护: loss 有限但梯度可能非有限(如 AAM sqrt 反传),
                # 直接 step 会用 NaN 梯度污染全部权重
                if not torch.isfinite(gnorm):
                    optimizer.zero_grad(set_to_none=True)
                    if rank == 0:
                        print(f"[{stage}] ep{epoch} step{step}: 梯度非有限, 跳过更新")
                else:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    gstep += 1
                    if args.scheduler == "cosine":
                        scheduler.step()
            for k, v in logs.items():
                run[k] = run.get(k, 0.0) + v
                win[k] = win.get(k, 0.0) + v
            run["loss"] = run.get("loss", 0.0) + loss.item() * args.grad_accum
            win["loss"] = win.get("loss", 0.0) + loss.item() * args.grad_accum
            nb += 1
            nw += 1
            if rank == 0 and step % args.log_interval == 0:
                lrs = [g["lr"] for g in optimizer.param_groups]
                msg = " ".join(f"{k}={v / nb:.4f}" for k, v in run.items())
                # 梯度冲突探针: 音素 vs 声纹 在编码器上的梯度范数与夹角余弦
                if stage == "audio":
                    probe = grad_conflict_probe(model, batch, cfg, grl_alpha, use_amp)
                    pmsg = " ".join(f"{k}={v:+.4f}" for k, v in probe.items())
                    msg = f"{msg} | {pmsg}"
                    if writer is not None:
                        for k, v in probe.items():
                            writer.add_scalar(f"grad/{k}", v, gstep)
                lr_str = (f"lr_enc={lrs[0]:.2e} lr_head={lrs[1]:.2e}"
                          if len(lrs) > 1 else f"lr={lrs[0]:.2e}")
                print(f"[{stage}] ep{epoch} {step}/{len(train_loader)} "
                      f"{msg} {lr_str}")
                if writer is not None:
                    for k, v in win.items():
                        writer.add_scalar(f"train/{k}", v / nw, gstep)
                    writer.add_scalar("train/lr", lrs[-1], gstep)
                    if len(lrs) > 1:
                        writer.add_scalar("train/lr_enc", lrs[0], gstep)
                win, nw = {}, 0
            if args.max_steps and step >= args.max_steps:
                break

        if skip_dev:
            # 跳过昂贵 dev 评估(过拟合自检/全量诊断提速), 直接盯 train 指标
            tr = {k: v / max(1, nb) for k, v in run.items()}
            watch_val = reduce_mean(tr.get(watch, tr.get("loss", 0.0)), world)
            improved = watch_val > best_val if mode == "max" else watch_val < best_val
            if improved:
                best_val, no_improve = watch_val, 0
            else:
                no_improve += 1
            if rank == 0:
                tag = "train(overfit)" if overfit else "train(no-dev)"
                msg = " ".join(f"{k}={v:.4f}" for k, v in tr.items())
                print(f"[{stage}] ep{epoch} {tag}: {msg} "
                      f"({time.time() - t0:.0f}s)")
                ckpt_meta = {**meta, "best_dev": best_val, "best_val": best_val,
                             "watch": watch}
                save_ckpt(save_dir / "last.pt", model, optimizer, scheduler, epoch,
                          cfg, ckpt_meta)
                if improved:
                    save_ckpt(save_dir / "best.pt", model, optimizer, scheduler,
                              epoch, cfg, ckpt_meta)
            if ddp:
                dist.barrier()
            if no_improve >= args.patience:
                if rank == 0:
                    print(f"[{stage}] overfit: 连续 {args.patience} 轮 {watch} 无提升, "
                          f"停于 ep{epoch} (best {watch}={best_val:.4f})")
                break
            continue
        # 评估统一用目标间隔, 避免 warmup 期间 dev loss 随 margin 变化而不可比
        unwrap(model).phoneme_head.set_margin(cfg.aam_margin)
        dev = evaluate(model, dev_loader, cfg, stage, device, use_amp=use_amp)
        dev_loss = reduce_mean(dev["loss"], world)
        watch_val = reduce_mean(dev.get(watch, dev["loss"]), world)
        if args.scheduler == "plateau":
            scheduler.step(dev_loss)
        improved = watch_val > best_val if mode == "max" else watch_val < best_val
        if improved:
            best_val = watch_val
            no_improve = 0
        else:
            no_improve += 1
        if rank == 0:
            msg = " ".join(f"{k}={v:.4f}" for k, v in dev.items())
            print(f"[{stage}] ep{epoch} dev: {msg} ({time.time() - t0:.0f}s)")
            if writer is not None:
                for k, v in dev.items():
                    writer.add_scalar(f"dev/{k}", v, epoch)
                writer.add_scalar(f"dev/best_{watch}", best_val, epoch)
            ckpt_meta = {**meta, "best_dev": best_val, "best_val": best_val,
                         "watch": watch}
            save_ckpt(save_dir / "last.pt", model, optimizer, scheduler, epoch,
                      cfg, ckpt_meta)
            if improved:
                save_ckpt(save_dir / "best.pt", model, optimizer, scheduler, epoch,
                          cfg, ckpt_meta)
                print(f"[{stage}] 新最优, 已保存 best.pt ({watch}={best_val:.4f})")
        if ddp:
            dist.barrier()
        # 平台早停: 连续 patience 轮 watch 指标无提升即结束本阶段(带最优权重进下一阶段)
        if no_improve >= args.patience:
            if rank == 0:
                print(f"[{stage}] 连续 {args.patience} 轮 {watch} 无提升, "
                      f"平台早停于 ep{epoch} (best {watch}={best_val:.4f})")
            break

    if writer is not None:
        writer.close()
    return save_dir / "best.pt"


def run_pipeline(args, cfg, rank: int, world: int, ddp: bool, device,
                 teacher: Synaspot = None):
    """四阶段课程式流水线: audio-only -> text-only -> multimodal.

    每阶段平台早停, 最优权重逐级传递, 先精通一块能力再练下一块.
    阶段1 音素预训练 (audio-only): margin=0 + 关对抗, 编码器先"精通"音素.
    阶段2 判别微调  (audio-only):  目标 margin + 封顶对抗 + 小lr. => audio-only 模型
    阶段3 text-only 对齐:          冻结音频, 逐帧余弦把文本嵌入拉进音频音素空间,
                                   解 InfoNCE 冷启动. => text-only 模型(含音频权重)
    阶段4 multimodal:              载入 text-only, InfoNCE 细粒度音文对齐.
    """
    import copy

    def banner(s):
        if rank == 0:
            print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70, flush=True)

    # ---- 阶段1: 音素预训练 (纯音素, margin=0, 关对抗) ----
    banner("[pipeline] 阶段 1/4: 音素预训练 (audio-only, margin=0, 关对抗)")
    cfg1 = copy.deepcopy(cfg)
    cfg1.aam_margin = 0.0
    a1 = copy.deepcopy(args)
    a1.epochs = args.stage_epochs
    a1.grl_cap = 0.0       # 关对抗
    a1.margin_ramp = 0.0   # margin 恒为0
    a1.audio_init = None
    best1 = train_stage("audio", a1, cfg1, rank, world, ddp, device,
                        init_ckpt=None, name="s1_ph", teacher=teacher)

    # ---- 阶段2: 判别微调 (加 margin + 封顶对抗, 小lr) => audio-only 模型 ----
    banner(f"[pipeline] 阶段 2/4: 判别微调 (audio-only, margin={cfg.aam_margin}, "
           f"grl_cap={args.grl_cap}, lr={args.finetune_lr})")
    a2 = copy.deepcopy(args)
    a2.epochs = args.stage_epochs
    a2.lr = args.finetune_lr
    best2 = train_stage("audio", a2, cfg, rank, world, ddp, device,
                        init_ckpt=best1, name="audio", teacher=teacher)

    # ---- 阶段3: text-only 对齐 (冻结音频, 逐帧余弦, 解 InfoNCE 冷启动) ----
    banner(f"[pipeline] 阶段 3/4: text-only 对齐 (冻结音频, 逐帧余弦, "
           f"align_cls_w={cfg.align_cls_w})")
    at = copy.deepcopy(args)
    at.epochs = args.stage_epochs
    # text_encoder 从零训, 需 base lr(1e-3); 音频已冻结, lr 只作用于文本分支.
    best_text = train_stage("text", at, cfg, rank, world, ddp, device,
                            init_ckpt=best2, name="text")

    # ---- 阶段4: multimodal 对比学习 (从 text-only 初始化, InfoNCE 已有好起点) ----
    banner("[pipeline] 阶段 4/4: multimodal 对比学习 (InfoNCE, 载入 text-only)")
    a3 = copy.deepcopy(args)
    a3.epochs = args.stage_epochs
    # 音频编码器联合微调(未冻结), 靠 L_ph(权重 alpha_M)保住音素特征;
    # 文本/混合分支已从 text-only 阶段对齐, InfoNCE 不再冷启动卡死.
    best3 = train_stage("mixed", a3, cfg, rank, world, ddp, device,
                        init_ckpt=best_text, name="mixed", teacher=teacher)

    if rank == 0:
        print(f"\n[pipeline] 四阶段完成, 最终模型: {best3}", flush=True)
    return best3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--save-dir", default=None,
                    help=f"检查点输出目录; 默认 checkpoints/<audio-backbone> "
                         f"(即 {DEFAULT_CKPT}/<dfsmn|ssl>)")
    ap.add_argument("--stage", choices=["audio", "text", "mixed", "both"],
                    default="both")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64, help="每卡批大小")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--enc-lr-mult", type=float, default=1.0,
                    help="编码器学习率倍数(与其余分支分离): 1=不分离; "
                         "编码器被多目标拉扯易饿死, 建议 3~5")
    ap.add_argument("--patience", type=int, default=8,
                    help="平台早停耐心轮数; 论文训50epoch, 默认8避免过早停")
    ap.add_argument("--scheduler", choices=["cosine", "plateau"], default="cosine",
                    help="cosine=warmup+余弦退火(按优化器步,推荐); "
                         "plateau=ReduceLROnPlateau(按dev,易过早塌缩)")
    ap.add_argument("--warmup-steps", type=int, default=1000,
                    help="cosine 调度的 warmup 优化器步数")
    ap.add_argument("--min-lr", type=float, default=1e-5,
                    help="cosine 调度的最低学习率")
    ap.add_argument("--grl-start", type=float, default=0.15,
                    help="对抗梯度反转开始施加的训练进度比例(之前为0, 纯学音素)")
    ap.add_argument("--grl-ramp", type=float, default=0.35,
                    help="grl_alpha 从0线性升到1所需训练进度比例")
    ap.add_argument("--margin-ramp", type=float, default=0.30,
                    help="AAM 角度间隔从0线性升到目标值所需训练进度比例"
                         "(margin warmup: 先学音素再加间隔)")
    ap.add_argument("--grl-cap", type=float, default=1.0,
                    help="grl_alpha 上限. 编码器已说话人无关后对抗过强会反压音素学习, "
                         "封顶以平衡(如 0.5)")
    ap.add_argument("--aam-margin", type=float, default=None,
                    help="覆盖目标 AAM 角度间隔(默认用 config 的 0.2). 音素卡局部极小可降为 0.1")
    ap.add_argument("--aam-scale", type=float, default=None,
                    help="覆盖 AAM 缩放 s(默认 config 的 30). s 过大易把小模型锁进塌缩解, 可试 8~16")
    ap.add_argument("--audio-backbone", choices=["dfsmn", "ssl"],
                    default="dfsmn",
                    help="dfsmn=80维fbank从零训; ssl=离线缓存HuBERT特征(768维,50Hz)冻结只训适配器"
                         "(需先跑 extract_ssl.py)")
    ap.add_argument("--feat-frame-rate", type=int, default=None,
                    choices=[25, 50, 100],
                    help="统一特征帧率(实验开关): 所有主干输出与帧标签都对齐到该帧率, "
                         "标签按 100/rate 降采样. 25/50 做帧率消融, 100=论文fbank基线. "
                         "默认用 config 的 50")
    ap.add_argument("--phoneme-granularity",
                    choices=["initial", "syllable", "syllable_tone"], default=None,
                    help="音素粒度(实验开关): initial=仅声母; syllable=声母+韵母(去调音节); "
                         "syllable_tone=声母+韵母+声调. 须与 04_build_dataset.py --granularity "
                         "一致; 默认读 dataset_info.json 的 phoneme_granularity")
    # ---- 离线→在线蒸馏 (教师/学生不同语音编码器) ----
    ap.add_argument("--distill-teacher-ckpt", default=None,
                    help="启用蒸馏: 教师检查点路径(用 --audio-backbone <教师主干> 预先训好). "
                         "学生用本命令的 --audio-backbone(因果, 如 dfsmn), 教师离线冻结")
    ap.add_argument("--distill-teacher-backbone", choices=["ssl"],
                    default="ssl",
                    help="教师主干=离线特征来源, 须与教师训练时一致(默认 ssl=HuBERT)")
    ap.add_argument("--distill-w-embed", type=float, default=None,
                    help="E_A 嵌入级蒸馏权重(逐帧余弦), 默认 config 的 1.0")
    ap.add_argument("--distill-w-logit", type=float, default=None,
                    help="Logit 级 KD 权重(音素软标签), 默认 config 的 1.0")
    ap.add_argument("--distill-temperature", type=float, default=None,
                    help="KD 温度(软化教师分布), 默认 config 的 2.0")
    ap.add_argument("--la-tau", type=float, default=0.0,
                    help="Logit Adjustment 温度(抗多数类塌缩): 0=关闭, 1=标准, "
                         "塌缩严重可 1.5~2.0. 给音素 logits 加 tau*log(类先验)")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="音素损失标签平滑(防多数类过度自信), 常与 --la-tau 联用, 如 0.1")
    ap.add_argument("--align-cls-w", type=float, default=None,
                    help="text-only 阶段共享音素分类器辅助损失权重(覆盖 config): "
                         "0=纯逐帧余弦对齐(推荐起点), 0.5~1.0=叠加蒸馏式对齐")
    ap.add_argument("--alpha-m", type=float, default=None,
                    help="覆盖 mixed 阶段 l_ph 权重(冻音频时设 0)")
    ap.add_argument("--beta-m", type=float, default=None,
                    help="覆盖 mixed 阶段 l_clat 权重")
    ap.add_argument("--gamma-m", type=float, default=None,
                    help="覆盖 mixed 阶段 l_clam(混合嵌入 InfoNCE)权重, 强化 e_m 重建")
    ap.add_argument("--freeze-audio-mixed", action="store_true",
                    help="mixed 阶段冻结 audio_encoder/phoneme_head/speaker_clf, "
                         "只训 text_encoder+mixed, 给 e_m 一个稳定的 e_a 重建目标")
    ap.add_argument("--audio-init", default=None,
                    help="audio 阶段初始化检查点(多阶段训练: 先 margin=0 预训练, 再载入做判别性微调)")
    ap.add_argument("--hidden-dim", type=int, default=None,
                    help="覆盖音频编码器隐藏维度(默认 config 256). 容量不足时扩容(如 512)")
    ap.add_argument("--proj-dim", type=int, default=None,
                    help="覆盖 DFSMN 瓶颈维度(默认 config 128), 未指定时随 hidden 同比扩")
    ap.add_argument("--pipeline", action="store_true",
                    help="三阶段自动流水线: 音素预训练->判别微调->混合, "
                         "每阶段平台早停并带最优权重进下一阶段")
    ap.add_argument("--stage-epochs", type=int, default=30,
                    help="流水线每阶段的最大轮数(达到或平台早停即进下一阶段)")
    ap.add_argument("--finetune-lr", type=float, default=3e-4,
                    help="流水线判别微调阶段(阶段2)的学习率")
    ap.add_argument("--max-frames", type=int, default=1600)
    ap.add_argument("--dither", type=float, default=0.0,
                    help="在线 fbank 抖动强度(kaldi dither). 本库 log-mel 特征值域小, "
                         "kaldi 默认 dither=1.0 的噪声与信号同量级会淹没特征(实测 diff std "
                         "3.3 vs 信号 std 3.2), dfsmn 主干从零训练应设 0; 有预训练可试 <0.1")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--log-interval", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="每 epoch 最多训练步数(调试用), 0=不限制")
    ap.add_argument("--train-limit", type=int, default=0,
                    help="过拟合自检: 只用前 N 条训练样本(并自动关 dither/shuffle). "
                         "判别架构/标签 bug(N 小且多轮后 train acc 仍≈随机=有 bug). 0=不限制")
    ap.add_argument("--no-dev-eval", action="store_true",
                    help="全量训练也跳过每轮 dev 评估(诊断/提速), 改盯 train 指标")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-tensorboard", action="store_true",
                    help="禁用 TensorBoard 日志")
    args = ap.parse_args()
    # 未显式指定时按主干分目录, 避免 dfsmn/ssl 实验结果互相覆盖
    if args.save_dir is None:
        args.save_dir = str(DEFAULT_CKPT / args.audio_backbone)

    rank, world, ddp = setup_ddp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)

    data_dir = Path(args.data_dir)
    with open(data_dir / "dataset_info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    cfg = ModelConfig(vocab_size=info["vocab_size"],
                      num_speakers=info["num_speakers_train"])
    cfg.audio_backbone = getattr(args, "audio_backbone", "dfsmn")
    # ---- 实验开关: 统一特征帧率 + 音素粒度 (训练之初配置) ----
    if args.feat_frame_rate is not None:
        cfg.feat_frame_rate = args.feat_frame_rate
    # 粒度: 优先命令行, 否则读 dataset_info.json (数据构建时落盘)
    gran = args.phoneme_granularity or info.get("phoneme_granularity")
    if gran:
        cfg.phoneme_granularity = gran
    # ---- 蒸馏开关与超参 (给了教师检查点即启用) ----
    if args.distill_teacher_ckpt:
        cfg.distill_enabled = True
        cfg.distill_teacher_ckpt = args.distill_teacher_ckpt
        cfg.distill_teacher_backbone = args.distill_teacher_backbone
    if args.distill_w_embed is not None:
        cfg.distill_w_embed = args.distill_w_embed
    if args.distill_w_logit is not None:
        cfg.distill_w_logit = args.distill_w_logit
    if args.distill_temperature is not None:
        cfg.distill_temperature = args.distill_temperature
    if args.aam_margin is not None:
        cfg.aam_margin = args.aam_margin
    if getattr(args, "aam_scale", None) is not None:
        cfg.aam_scale = args.aam_scale
    if args.align_cls_w is not None:
        cfg.align_cls_w = args.align_cls_w
    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim
        if args.proj_dim is None:          # 瓶颈维度随隐藏维度同比扩
            cfg.proj_dim = args.hidden_dim // 2
    if args.proj_dim is not None:
        cfg.proj_dim = args.proj_dim
    if args.alpha_m is not None:
        cfg.alpha_M = args.alpha_m
    if args.beta_m is not None:
        cfg.beta_M = args.beta_m
    if args.gamma_m is not None:
        cfg.gamma_M = args.gamma_m

    # 训练之初一致性校验: 帧率×主干可整除重采样, 粒度与数据集词表自洽, 特征已提取
    validate_config(cfg, info, data_dir, rank)

    if rank == 0:
        print(f"数据: {data_dir}, 词表 {cfg.vocab_size}, 说话人 {cfg.num_speakers}, "
              f"world_size={world}")

    # Logit Adjustment 先验: rank0 预计算并落缓存(首跑约几分钟), 其余 rank 等 barrier 后读缓存
    if getattr(args, "la_tau", 0.0) > 0:
        if rank == 0:
            get_log_priors(data_dir, cfg.vocab_size)
        if ddp:
            dist.barrier()

    # 蒸馏: 载入冻结的离线教师 (每个 rank 各载一份; 配置兼容性在 load_teacher 校验)
    teacher = load_teacher(cfg, device, rank) if cfg.distill_enabled else None

    if args.pipeline:
        run_pipeline(args, cfg, rank, world, ddp, device, teacher=teacher)
    else:
        if args.stage in ("audio", "both"):
            audio_init = Path(args.audio_init) if args.audio_init else None
            best_audio = train_stage("audio", args, cfg, rank, world, ddp, device,
                                     init_ckpt=audio_init, teacher=teacher)
        if args.stage == "text":
            # 单独跑 text-only 对齐: 需先有 audio-only 模型(audio_init 或 exp/audio/best.pt)
            # 音频编码器冻结, 不做蒸馏, 故不传 teacher
            init = Path(args.audio_init) if args.audio_init else \
                (Path(args.save_dir) / "audio" / "best.pt")
            train_stage("text", args, cfg, rank, world, ddp, device, init_ckpt=init)
        if args.stage in ("mixed", "both"):
            if args.stage == "mixed":
                # 允许用 --audio-init 指定初始化(如 text-only 检查点), 否则回退 audio/best.pt
                init = Path(args.audio_init) if args.audio_init else \
                    (Path(args.save_dir) / "audio" / "best.pt")
            else:
                init = best_audio
            train_stage("mixed", args, cfg, rank, world, ddp, device,
                        init_ckpt=init, teacher=teacher)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
