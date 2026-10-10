"""SYNASPOT 流式注册与解码 (论文 2.2).

流程:
1. enroll: 从注册音频/文本计算并缓存三种注册嵌入 E_A / E_T / E_M (L2 归一化)
2. feed:   流式送入音频块, 因果编码器对缓冲音频提取帧级嵌入 E_W,
           按 Function 6/7 计算每帧关键词得分:
             p'_ij : 因果滑动平均平滑 (窗口 w_smooth)
             Score : 打分窗口 w_scoring 内各单元最大相似度的几何平均
           Score_All = α_S·Score_A + β_S·Score_T + γ_S·Score_M

编码器为因果结构(rorder=0), 历史帧嵌入不随后续输入变化.
"""
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from pypinyin import Style, pinyin
from torchaudio.compliance import kaldi

from .config import ModelConfig
from .dataset import load_wav, resample_time
from .model import Synaspot


class SynaspotDecoder:
    def __init__(self, ckpt_path: str, device: str = "cuda",
                 w_smooth: int = None, w_scoring: int = None,
                 ssl_model_dir: str = None):
        ck = torch.load(ckpt_path, map_location="cpu")
        raw_cfg = ck["config"]
        self.cfg = ModelConfig.from_dict(raw_cfg)
        # 旧检查点无 feat_frame_rate: 当年 dfsmn=100Hz, ssl=50Hz, 按主干回填以保旧行为
        if "feat_frame_rate" not in raw_cfg:
            self.cfg.feat_frame_rate = 50 if self.cfg.audio_backbone == "ssl" else 100
        # 窗口按统一帧率自适应: w_smooth/w_scoring/enroll_expand_frames 以 100Hz 为参考,
        # 特征对齐到 feat_frame_rate 后等比缩小, 保持物理时长不变(50Hz -> 减半).
        ds = self.cfg.label_ds
        if ds > 1:
            if w_smooth is None:
                self.cfg.w_smooth = max(1, round(self.cfg.w_smooth / ds))
            if w_scoring is None:
                self.cfg.w_scoring = max(1, round(self.cfg.w_scoring / ds))
            self.cfg.enroll_expand_frames = max(
                1, round(self.cfg.enroll_expand_frames / ds))
        if w_smooth is not None:
            self.cfg.w_smooth = w_smooth
        if w_scoring is not None:
            self.cfg.w_scoring = w_scoring
        self.vocab = ck["vocab"]
        self.device = torch.device(device)
        self.model = Synaspot(self.cfg).to(self.device).eval()
        # log_priors 是注入缓冲(空=未启用, 27=已启用), 载入时形状可能不一致,
        # 跳过它, 由下方 set_log_priors (训练用)或直接用原始 logits(推理无需 LA)处理.
        sd = {k: v for k, v in ck["model"].items()
              if "log_priors" not in k and "class_weight" not in k}
        self.model.load_state_dict(sd, strict=False)
        # ssl 主干: 载入冻结的 HuBERT 用于把波形提成 768 维帧特征(50Hz).
        # HuBERT 双向非因果, 只能离线整段提特征(评测/注册), 不做真流式.
        self.ssl_model = None
        if getattr(self.cfg, "audio_backbone", "dfsmn") == "ssl":
            from transformers import HubertModel
            ssl_dir = (Path(ssl_model_dir) if ssl_model_dir
                       else Path(__file__).parent / "Model_File" / "chinese-hubert-base")
            self.ssl_model = HubertModel.from_pretrained(str(ssl_dir)) \
                .to(self.device).eval()
            for p in self.ssl_model.parameters():
                p.requires_grad_(False)
        self.reset()
        self.ea = self.et = self.em = None

    # ---------------- 注册 ----------------
    def _fbank(self, wav: torch.Tensor) -> torch.Tensor:
        """音频特征入口: 按主干提特征并重采样到 cfg.feat_frame_rate."""
        bb = getattr(self.cfg, "audio_backbone", "dfsmn")
        if bb == "ssl":
            return resample_time(self._ssl_feats(wav), 50, self.cfg.feat_frame_rate)
        feats = kaldi.fbank(wav, num_mel_bins=self.cfg.num_fbanks,
                            frame_length=self.cfg.frame_length_ms,
                            frame_shift=self.cfg.frame_shift_ms,
                            sample_frequency=self.cfg.sample_rate, dither=0.0)
        if self.cfg.use_delta:
            from torchaudio.functional import compute_deltas
            d = compute_deltas(feats)
            dd = compute_deltas(d)
            feats = torch.cat([feats, d, dd], dim=-1)
        feats = feats - feats.mean(dim=0)
        return resample_time(feats, 100, self.cfg.feat_frame_rate)

    @torch.no_grad()
    def _ssl_feats(self, wav: torch.Tensor) -> torch.Tensor:
        """波形 -> HuBERT last_hidden_state (T, 768) @50Hz. 逐句归一化, 与离线提取一致."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        x = wav.to(self.device).float()
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-7)
        return self.ssl_model(x).last_hidden_state[0]

    @torch.no_grad()
    def enroll(self, audio_path: str = None, text: str = None):
        """audio_path: 注册音频; text: 关键词文本(中文). 至少提供其一."""
        cfg = self.cfg
        if audio_path is not None:
            wav = load_wav(audio_path, cfg.sample_rate)
            feats = self._fbank(wav).unsqueeze(0).to(self.device)
            self.ea = F.normalize(self.model.encode_audio(feats)[0], dim=-1)
        else:
            self.ea = None

        if text is not None:
            ids = self.text_to_ids(text)
            if not ids:
                raise ValueError(f"文本无法映射到拼音词表: {text}")
            # 每个拼音 token 展开 enroll_expand_frames 帧, 与训练时帧级输入一致
            frame_ids = torch.tensor(ids, device=self.device).repeat_interleave(
                cfg.enroll_expand_frames).unsqueeze(0)
            self.et = F.normalize(self.model.encode_text(frame_ids)[0], dim=-1)
        else:
            self.et = None

        if self.ea is not None and self.et is not None:
            # key/value 为 E_A, mask 长度须与音频帧数一致
            mask = torch.ones(1, self.ea.size(0), dtype=torch.bool,
                              device=self.device)
            em = self.model.encode_mixed(self.et.unsqueeze(0), self.ea.unsqueeze(0),
                                         mask)
            self.em = F.normalize(em[0], dim=-1)
        else:
            self.em = None
        return self

    def text_to_ids(self, text: str):
        # 粒度由配置显式给出(训练时从 dataset_info 落盘到 ckpt), 不再靠词表启发式
        gran = getattr(self.cfg, "phoneme_granularity", None)
        if gran == "initial":
            use_tone, use_initial = False, True
        elif gran == "syllable":
            use_tone, use_initial = False, False
        elif gran == "syllable_tone":
            use_tone, use_initial = True, False
        else:
            # 旧检查点无粒度字段: 回退到词表启发式反推
            special = ("<PAD>", "<UNK>", "<SIL>")
            real = [t for t in self.vocab if t not in special]
            use_tone = any(t[-1:].isdigit() for t in real)
            initials = {"b","p","m","f","d","t","n","l","g","k","h","j","q","x",
                        "zh","ch","sh","r","z","c","s","y","w","<O>"}
            use_initial = len(real) < 30 and len(set(real) & initials) > len(real) // 2
        ids = []
        for py in pinyin(text, style=Style.TONE3, heteronym=False):
            token = py[0]
            if not use_tone:
                token = re.sub(r"[1-5]$", "", token)
            if use_initial:
                if token not in self.vocab and token != "<O>":
                    for pre in ("zh","ch","sh","b","p","m","f","d","t","n","l",
                                "g","k","h","j","q","x","r","z","c","s","y","w"):
                        if token.startswith(pre):
                            token = pre
                            break
                    else:
                        token = "<O>"
            if token in self.vocab:
                ids.append(self.vocab[token])
            elif self.vocab.get("<UNK>") is not None:
                ids.append(self.vocab["<UNK>"])
        return ids

    # ---------------- 流式解码 ----------------
    def reset(self):
        self.buffer = torch.zeros(1, 0)
        self.n_scored = 0

    @torch.no_grad()
    def feed(self, wav: torch.Tensor) -> dict:
        """送入 16kHz 单声道波形 (1, N) 或 (N,), 返回新增帧的得分."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        self.buffer = torch.cat([self.buffer, wav.cpu()], dim=1)
        if self.buffer.size(1) < 400:
            return {"score": torch.zeros(0)}
        feats = self._fbank(self.buffer).unsqueeze(0).to(self.device)
        e_w = F.normalize(self.model.encode_audio(feats)[0], dim=-1)  # (T, H)

        out = {}
        for name, enroll in (("A", self.ea), ("T", self.et), ("M", self.em)):
            if enroll is None:
                continue
            s = self._modality_scores(enroll, e_w)
            out[f"score_{name}"] = s[self.n_scored:].cpu()
        if out:
            keys = list(out.keys())
            n_new = out[keys[0]].numel()
            fused = sum(self._weight(k) * out[k] for k in keys) / \
                sum(self._weight(k) for k in keys)
            out["score"] = fused
            self.n_scored += n_new
        else:
            out["score"] = torch.zeros(0)
        return out

    def _weight(self, key: str) -> float:
        return {"score_A": self.cfg.alpha_S, "score_T": self.cfg.beta_S,
                "score_M": self.cfg.gamma_S}[key]

    def _modality_scores(self, enroll: torch.Tensor, e_w: torch.Tensor) -> torch.Tensor:
        """Function 6 + 7. enroll: (n, H), e_w: (T, H), 均已 L2 归一化."""
        cfg = self.cfg
        P = enroll @ e_w.t()  # (n, T)
        # Function 6: 因果滑动平均
        cs = torch.cumsum(P, dim=1)
        T = P.size(1)
        idx = torch.arange(T, device=P.device)
        h = torch.clamp_min(idx - cfg.w_smooth + 1, 0)
        prev = torch.where(h > 0, cs[:, torch.clamp(h - 1, min=0)], 0.0)
        P_smooth = (cs - prev) / (idx - h + 1).float().unsqueeze(0)
        # Function 7: 打分窗口内取每单元最大值
        pad = torch.full((P.size(0), cfg.w_scoring - 1), -1e9, device=P.device)
        win = torch.cat([pad, P_smooth], dim=1).unfold(1, cfg.w_scoring, 1)
        M = win.max(dim=-1).values  # (n, T)
        # 连乘(几何平均): 论文取前 n-1 个单元
        n = enroll.size(0)
        k = n - 1 if (cfg.score_skip_last and n > 1) else n
        M = torch.clamp(M[:k], min=1e-6)
        return torch.exp(torch.log(M).mean(dim=0))

    # ---------------- 离线便捷接口 ----------------
    @torch.no_grad()
    def detect(self, wav_path: str, chunk_frames: int = 20) -> dict:
        """对整段音频按块流式解码, 返回完整得分序列."""
        wav = load_wav(wav_path, self.cfg.sample_rate)
        self.reset()
        scores = []
        chunk = chunk_frames * 160  # 10ms/帧
        for i in range(0, wav.size(1), chunk):
            out = self.feed(wav[:, i:i + chunk])
            scores.append(out["score"])
        return {"score": torch.cat(scores) if scores else torch.zeros(0),
                "num_frames": self.n_scored}


def load_manifest_records(manifest_path: str):
    with open(manifest_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    # 简单自测: 用一条训练样本同时做音频+文本注册, 再流式回放该样本
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dec = SynaspotDecoder(args.ckpt, device=args.device)
    dec.enroll(audio_path=args.wav, text=args.text)
    res = dec.detect(args.wav)
    s = res["score"]
    print(f"帧数={res['num_frames']}, 得分: min={s.min():.4f} "
          f"max={s.max():.4f} mean={s.mean():.4f}")
    print("得分轨迹(每10帧):", [f"{v:.3f}" for v in s[::10].tolist()])
