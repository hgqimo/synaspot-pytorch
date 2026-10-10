"""SYNASPOT 模型.

组件:
- AudioEncoder : L 层 DFSMN, 输入 FBank, 输出帧级音频嵌入 E_A (论文 2.1.1)
- SpeakerClassifier : attentive pooling + 线性层, 配合梯度反转去除声纹 (论文 2.1.1)
- TextEncoder : Embedding + LSTM, 输出文本嵌入 E_T (论文 2.1.2)
- MixedEmbedding : cross-attention(Q=E_T, K/V=E_A), 输出混合嵌入 E_M (论文 2.1.2)

训练时文本分支输入为帧级展开的拼音 id 序列(即帧标签), 与 E_A 逐帧对齐,
用于 Function 3/4 的帧级对比学习; 注册推理时每个拼音 token 展开
cfg.enroll_expand_frames 帧, 保持与训练一致的输入形态.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .losses import AAMSoftmax


class MemoryBlock(nn.Module):
    """FSMN memory block: 时间维深度卷积, 支持因果(lorder)与前看(rorder)."""

    def __init__(self, dim: int, lorder: int, rorder: int = 0):
        super().__init__()
        self.lorder = lorder
        self.rorder = rorder
        self.conv = nn.Conv1d(dim, dim, kernel_size=lorder + rorder + 1, groups=dim)
        nn.init.zeros_(self.conv.bias)
        # 关键: 深度卷积默认初始化方差较大, kernel=21 时会等效于移动平均,
        # 把相邻帧压成相似向量, 经多层后编码器输出有效秩塌缩到个位数
        # (实测 256 维 -> 秩 6), 音素学习无从展开. 用小权重让 memory 初始近零,
        # 残差路径(x)主导, 初始等效近似恒等, 训练再逐步学时序平滑.
        nn.init.normal_(self.conv.weight, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        y = x.transpose(1, 2)
        y = F.pad(y, (self.lorder, self.rorder))
        return self.conv(y).transpose(1, 2)


class DFSMNLayer(nn.Module):
    """单层 DFSMN: 下投影 -> memory -> 上投影, 残差 + LayerNorm."""

    def __init__(self, hidden_dim: int, proj_dim: int, lorder: int, rorder: int,
                 dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(hidden_dim, proj_dim)
        self.memory = MemoryBlock(proj_dim, lorder, rorder)
        self.up = nn.Linear(proj_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        h = self.memory(h)
        h = self.up(h)
        return self.norm(x + self.dropout(h))


class AudioEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # 输入维度随主干: dfsmn=80 fbank, ssl=768
        self.input_proj = nn.Linear(cfg.feat_dim, cfg.hidden_dim)
        self.layers = nn.ModuleList([
            DFSMNLayer(cfg.hidden_dim, cfg.proj_dim, cfg.lorder, cfg.rorder,
                       cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        """fbank/ssl 特征: (B, T, F) -> E_A: (B, T, H). 因果结构, 可流式."""
        x = self.input_proj(fbank)
        for layer in self.layers:
            x = layer(x)
        return x


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class AttentivePooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        w = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * w).sum(dim=1)


class SpeakerClassifier(nn.Module):
    """attentive pooling + 线性分类器 (论文 2.1.1, 训练时经梯度反转)."""

    def __init__(self, dim: int, num_speakers: int):
        super().__init__()
        self.pool = AttentivePooling(dim)
        self.fc = nn.Linear(dim, num_speakers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                grl_alpha: float = 1.0) -> torch.Tensor:
        pooled = self.pool(x, mask)
        pooled = GradientReversal.apply(pooled, grl_alpha)
        return self.fc(pooled)


class TextEncoder(nn.Module):
    """Embedding + LSTM, 输入帧级拼音 id 序列, 输出帧级 E_T."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.text_embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(cfg.text_embed_dim, cfg.text_lstm_hidden,
                            num_layers=cfg.text_lstm_layers, batch_first=True)
        self.proj = (nn.Identity() if cfg.text_lstm_hidden == cfg.hidden_dim
                     else nn.Linear(cfg.text_lstm_hidden, cfg.hidden_dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) 帧级拼音 id -> E_T: (B, T, H)."""
        x = self.embed(ids)
        out, _ = self.lstm(x)
        # 残差跳连: 保留 embedding 的 token 判别性(27个近正交方向),
        # 否则 1 层 LSTM 会把输出压成 ~1.5 维(实测跨关键词秩 1.5/256),
        # 致对比学习永远卡在随机(EER 50%), 无论怎么调 lr/冻编码器都救不回来.
        return self.proj(out + x)


class MixedEmbedding(nn.Module):
    """cross-attention: Q=E_T, K/V=E_A (论文 2.1.2)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads,
                                          dropout=cfg.dropout, batch_first=True)

    def forward(self, e_t: torch.Tensor, e_a: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        em, _ = self.attn(query=e_t, key=e_a, value=e_a,
                          key_padding_mask=~mask, need_weights=False)
        return em


class Synaspot(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.audio_encoder = AudioEncoder(cfg)
        self.phoneme_head = AAMSoftmax(cfg.hidden_dim, cfg.vocab_size,
                                       cfg.aam_scale, cfg.aam_margin)
        self.speaker_clf = SpeakerClassifier(cfg.hidden_dim, cfg.num_speakers)
        self.text_encoder = TextEncoder(cfg)
        self.mixed = MixedEmbedding(cfg)

    def encode_audio(self, fbank: torch.Tensor) -> torch.Tensor:
        return self.audio_encoder(fbank)

    def encode_text(self, frame_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(frame_ids)

    def encode_mixed(self, e_t: torch.Tensor, e_a: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        return self.mixed(e_t, e_a, mask)

    def speaker_logits(self, e_a: torch.Tensor, mask: torch.Tensor,
                       grl_alpha: float = 1.0) -> torch.Tensor:
        return self.speaker_clf(e_a, mask, grl_alpha)

    def phoneme_logits(self, e_a: torch.Tensor,
                       targets: torch.Tensor = None) -> torch.Tensor:
        return self.phoneme_head(e_a, targets)

    def forward(self, fbank: torch.Tensor, labels: torch.Tensor = None,
                mask: torch.Tensor = None, stage: str = "audio",
                grl_alpha: float = 1.0) -> dict:
        """统一前向入口(DDP 要求所有计算经 forward 触发).

        stage="audio": 返回 e_a, spk_logits
        stage="text":  返回 e_a, e_t (text-only 对齐, 不走 mixed)
        stage="mixed": 返回 e_a, e_t, e_m
        grl_alpha: 梯度反转强度, 训练中从 0 渐进升到 1 (先学音素再抗声纹).
        """
        e_a = self.audio_encoder(fbank)
        out = {"e_a": e_a}
        if stage == "audio":
            if mask is not None:
                out["spk_logits"] = self.speaker_clf(e_a, mask, grl_alpha)
        else:
            e_t = self.text_encoder(labels)
            out["e_t"] = e_t
            if stage == "mixed":
                # 与注册(decode.py)保持一致: 归一化后再进 cross-attention.
                # 否则训练用原始幅值(注意力尖锐)、注册用归一化(注意力近均匀),
                # 注册时 e_m 被糊成 e_a 帧均值, M 模态判别力崩坏(M-only EER 15.6%).
                e_m = self.mixed(F.normalize(e_t, dim=-1),
                                 F.normalize(e_a, dim=-1), mask)
                out["e_m"] = e_m
        return out
