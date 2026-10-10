"""SYNASPOT 损失函数.

- AAMSoftmax: 加性角度间隔损失 (论文 Function 1), 用于帧级音素分类
- contrastive_loss: 帧级 InfoNCE 对比学习损失 (论文 Function 3/4)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    """加性角度间隔 (AAM) 分类头 (论文 Function 1).

    L = -log( exp(s*cos(theta_y + m)) / (exp(s*cos(theta_y + m)) + Σ_{j≠y} exp(s*cos(theta_j))) )

    严格按论文公式: 目标类 cos 用 cos(theta + m) = cos*cos_m - sin*sin_m,
    非目标类用 cos(theta). 不加 ArcFace 原版的 where(cos>th, phi, cos-mm) 兜底分支——
    该分支在 cos≈0(多类未训练) 时把目标 logit 压成 s*(cos-mm) 负值, 致训练崩塌.
    """

    def __init__(self, in_features: int, num_classes: int,
                 scale: float = 30.0, margin: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.s = scale
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.set_margin(margin)
        # 抗长尾塌缩开关(默认关): Logit Adjustment 先验 + 标签平滑 + CB类权重, 由 set_log_priors 注入
        self.register_buffer("log_priors", torch.empty(0))
        self.register_buffer("class_weight", torch.empty(0))
        self.la_tau = 0.0
        self.label_smoothing = 0.0

    def set_log_priors(self, log_priors: torch.Tensor, tau: float = 1.0,
                       label_smoothing: float = 0.0):
        """启用 Logit Adjustment (Menon et al. 2021) 抗多数类塌缩.

        训练时给 logits 加 tau*log(p_c): tau=1 时最优解为与先验无关的似然
        log p(x|c), 即逼模型学"平衡分类器", 不再退化到全猜高频类.
        塌缩严重可把 tau 调到 1.5~2.0 加大纠偏力度.
        """
        self.log_priors = log_priors.detach().float()  # 更新 __init__ 已注册的空缓冲
        self.la_tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def set_margin(self, margin: float):
        """动态设置角度间隔. 训练中从 0 渐进升到目标值: 先让音素快速收敛,
        再施加间隔增强判别性."""
        self.m = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        """x: (..., H). targets 为 None 时返回 s*cos(theta) (注册/推理用).

        强制 fp32 计算: bf16 下 cos→1 时 sqrt(1-cos²) 的反传梯度为 inf,
        会一步污染全部参数 (训练 NaN 的根源).
        """
        with torch.autocast("cuda", enabled=False):
            x = x.float()
            w = self.weight.float()
            cos = F.linear(F.normalize(x, dim=-1), F.normalize(w, dim=-1))
            if targets is None:
                return cos * self.s
            # 论文 Function 1: 目标类 cos(theta_y + m), 非目标类 cos(theta_j)
            sin = torch.sqrt((1.0 - cos.pow(2)).clamp(min=1e-12))
            phi = cos * self.cos_m - sin * self.sin_m  # cos(theta + m)
            one_hot = F.one_hot(targets, self.num_classes).to(cos.dtype)
            logits = (one_hot * phi + (1.0 - one_hot) * cos) * self.s
            return logits


# 音素分类需排除的特殊类: <PAD>=0, <UNK>=1, <SIL>=2.
# <SIL> 占帧数 ~30% 且是最大单类: 若参与损失, 模型会塌缩到"全预测静音",
# 使 dev acc 假性停在 ≈<SIL> 占比(0.277)而音素准确率为 0.
PH_IGNORE = (0, 1, 2)


def phoneme_loss(aam: AAMSoftmax, emb: torch.Tensor, labels: torch.Tensor,
                 mask: torch.Tensor):
    """帧级音素 AAM 损失. emb/labels/mask: (B,T,H)/(B,T)/(B,T).

    只在真实音素帧上计算(排除 <PAD>/<UNK>/<SIL>), 避免被占比 ~30% 的
    <SIL> 主导而塌缩; 返回的 acc 也仅统计音素帧, 才是真实音素准确率.
    """
    emb_v = emb[mask]
    lab_v = labels[mask]
    keep = torch.ones_like(lab_v, dtype=torch.bool)
    for c in PH_IGNORE:
        keep &= lab_v != c
    emb_v, lab_v = emb_v[keep], lab_v[keep]
    if lab_v.numel() == 0:
        zero = emb.new_zeros(())
        return zero, zero
    logits = aam(emb_v, lab_v)
    # Logit Adjustment: 给 logits 加 tau*log(p_c), 抵消多数类先验优势.
    # 注意: 训练损失(LA 后)与 acc 必须用同一套 logits, 否则 acc 统计的是
    # "未优化的原始 logits", LA 把高频类压低后原始 argmax 会被噪声主导,
    # 出现 l_ph 在降但 acc_ph 卡 0 的假象.
    lp = getattr(aam, "log_priors", None)
    la_on = lp is not None and lp.numel() > 0 and aam.la_tau > 0
    if la_on:
        logits = logits + aam.la_tau * lp.to(logits.device)
    # 标签平滑: 防多数类过度自信, 与 Logit Adjustment 叠加抗塌缩
    ls = getattr(aam, "label_smoothing", 0.0)
    loss = F.cross_entropy(logits, lab_v, label_smoothing=ls)
    # acc 必须在"无 targets / 无 margin"的干净 logits 上统计: aam(emb, None)=s*cos.
    # 绝不能复用上面带 targets 的 logits——那里目标类被按真实标签施加了角度间隔,
    # 嵌入塌缩(所有帧同向)时该目标条件化 + LA 先验会把 acc 虚抬到 ~0.5
    # (实测塌缩模型: 训练同款 acc=0.50 vs 干净 acc=0.06), 使早停把"最塌缩"的
    # 权重当最优保存, 形成奖励塌缩的恶性循环. 干净 logits 才反映真实排序能力.
    # (LA 仍要加: 它是解码的一部分, 与训练目标一致)
    with torch.no_grad():
        logits_acc = aam(emb_v, None)
        if la_on:
            logits_acc = logits_acc + aam.la_tau * lp.to(logits_acc.device)
        logits_acc = logits_acc.clone()
        logits_acc[:, list(PH_IGNORE)] = float("-inf")
        acc = (logits_acc.argmax(dim=-1) == lab_v).float().mean()
    return loss, acc


def contrastive_loss(e_src: torch.Tensor, e_tgt: torch.Tensor, mask: torch.Tensor,
                     temperature: float, labels: torch.Tensor = None,
                     max_frames: int = 4096):
    """帧级 InfoNCE (论文 Function 3/4).

    正样本: 同一帧的 (e_src^i, e_tgt^i); 负样本: 其余有效帧.
    labels 提供时, 同拼音标签的帧不作为负样本(避免假负例).
    帧数超过 max_frames 时随机下采样以控制显存.
    """
    src = e_src[mask]
    tgt = e_tgt[mask]
    n = src.size(0)
    if n < 2:
        return src.new_zeros(())
    if n > max_frames:
        idx = torch.randperm(n, device=src.device)[:max_frames]
        src, tgt = src[idx], tgt[idx]
        n = max_frames
        lab = labels[mask][idx] if labels is not None else None
    else:
        lab = labels[mask] if labels is not None else None

    src = F.normalize(src, dim=-1)
    tgt = F.normalize(tgt, dim=-1)
    logits = (src @ tgt.t() / temperature).float()  # (N, N), CE 用 fp32 更稳

    if lab is not None:
        same = lab.unsqueeze(0) == lab.unsqueeze(1)
        same = same & ~torch.eye(n, dtype=torch.bool, device=src.device)
        logits = logits.masked_fill(same, float("-inf"))

    targets = torch.arange(n, device=src.device)
    return F.cross_entropy(logits, targets)


def logit_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                  temperature: float = 2.0) -> torch.Tensor:
    """Logit 级知识蒸馏 (KD): KL(softmax(t/T) || softmax(s/T)) * T^2.

    student/teacher_logits: (N, V) 干净 logits (AAMSoftmax 传 targets=None 的 s*cos,
    无 margin/无 LA, 才是可迁移的"教师决策"). teacher 已 detach/no_grad.
    温度 T 软化教师分布、放大类间差异信号; 乘 T^2 保持梯度量级与 T=1 一致.
    """
    if student_logits.size(0) == 0:
        return student_logits.new_zeros(())
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.detach().float()
    log_p_s = F.log_softmax(student_logits / temperature, dim=-1)
    p_t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2)


def frame_align_loss(e_src: torch.Tensor, e_tgt: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """逐帧余弦对齐: mean(1 - cos(e_src, e_tgt)) 在有效帧上.

    用于 text-only 阶段把文本嵌入直接拉向(冻结的)音频嵌入, 解决 InfoNCE 的
    冷启动问题: 随机初始化时音文嵌入正交, InfoNCE 梯度推不动、卡在 log(N);
    本损失只有正样本、无负样本, 梯度强且方向明确, 能先把文本编码器"搬进"
    音频音素空间, 之后 InfoNCE 才有好的起点做细粒度判别.

    目标嵌入 e_tgt 应来自冻结的音频编码器(调用方 detach), 只优化 e_src.
    因 e_tgt 冻结且音素判别(不同音素 -> 不同 e_tgt), e_src 逐 token 收敛到
    各自音素的平均音频嵌入, 不会塌缩.
    """
    src = e_src[mask].float()
    tgt = e_tgt[mask].float()
    if src.size(0) == 0:
        return e_src.new_zeros(())
    src = F.normalize(src, dim=-1)
    tgt = F.normalize(tgt, dim=-1)
    return (1.0 - (src * tgt).sum(dim=-1)).mean()
