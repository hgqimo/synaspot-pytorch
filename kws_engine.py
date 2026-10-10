"""SYNASPOT 流式关键词检测引擎 (可复用核心, 不依赖 Web 框架).

用于全双工语音对话系统的"打断(barge-in)"场景: 替代常开 ASR + 热词的后台识别,
用轻量关键词向量做流式匹配, 命中关键词即触发打断回调.

组成:
- KeywordStore      : 离线注册向量的本地持久化 (文本 / 音频注册 -> 向量落盘)
- KWSEngine         : 持有共享 SynaspotDecoder, 提供注册 / 列表 / 删除 / 开检测会话
- DetectionSession  : 一次流式检测会话, 吞入 16kHz PCM, 滑窗打分, 超阈值即触发

典型用法见 kws_server.py, 也可直接引用本模块集成进自有全双工系统:

    engine = KWSEngine(ckpt_path=".../exp/mixed/best.pt", store_dir="keyword_store",
                       ssl_model_dir=".../chinese-hubert-base")
    kw = engine.register("打断", text="停一下")
    ses = engine.new_session(kw["id"])
    while 收麦克风 PCM:
        score, triggered = ses.feed_pcm(pcm_bytes)
        if triggered:
            触发打断; ses.reset()  # 或保持冷却自动抑制
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from synaspot.decode import SynaspotDecoder

logger = logging.getLogger("synaspot.kws")


def vad_trim(wav: torch.Tensor, sample_rate: int, ratio: float = 0.2,
             pad_sec: float = 0.05, min_sec: float = 0.3):
    """按能量裁剪首尾静音, 返回 (裁剪后波形, 时长秒).

    注册音频必须紧凑, 原因有两点:
    1. 打分公式对全部注册单元 n 取几何平均, 静音/冗余帧会因无法对齐而拖垮得分;
    2. HuBERT 逐句归一化, 检测滑窗长度需与注册音频时长接近, 统计才匹配.
    """
    w = wav[0] if wav.dim() == 2 else wav
    fl = max(1, int(sample_rate * 0.02))
    if w.numel() < fl * 2:
        return wav, w.numel() / sample_rate
    rms = torch.stack([(w[i:i + fl] ** 2).mean().sqrt()
                       for i in range(0, w.numel() - fl, fl)])
    thr = rms.max() * ratio
    idx = (rms > thr).nonzero().flatten()
    if idx.numel() == 0:
        return wav, w.numel() / sample_rate
    pad = int(sample_rate * pad_sec)
    s = max(0, int(idx[0] * fl) - pad)
    e = min(w.numel(), int((idx[-1] + 1) * fl) + pad)
    if (e - s) < int(sample_rate * min_sec):  # 过短则向两侧补齐
        need = int(sample_rate * min_sec) - (e - s)
        s = max(0, s - need // 2)
        e = min(w.numel(), e + need - need // 2)
    seg = w[s:e]
    if wav.dim() == 2:
        seg = seg.unsqueeze(0)
    return seg, (e - s) / sample_rate


class KeywordStore:
    """关键词注册向量的本地持久化.

    磁盘布局:
        <store_dir>/index.json          元信息索引 (id/名称/文本/阈值/时间)
        <store_dir>/<id>.pt             注册嵌入 {ea, et, em} (各为 (n,H) 或 None)
    """

    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "index.json"
        self._index: dict = {}
        if self.index_path.exists():
            try:
                self._index = json.loads(self.index_path.read_text("utf-8"))
            except Exception as e:  # 索引损坏时从空开始
                logger.warning(f"index.json 读取失败, 重置索引: {e}")
                self._index = {}

    # ---- 写 ----
    def save(self, name: str, embeddings: dict, text: Optional[str] = None,
             threshold: float = 0.5, audio_name: Optional[str] = None,
             extra: Optional[dict] = None) -> dict:
        kid = uuid.uuid4().hex[:12]
        emb_file = self.store_dir / f"{kid}.pt"
        torch.save({
            "ea": embeddings.get("ea"),
            "et": embeddings.get("et"),
            "em": embeddings.get("em"),
        }, emb_file)
        meta = {
            "id": kid,
            "name": name,
            "text": text,
            "threshold": float(threshold),
            "audio_name": audio_name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_file": emb_file.name,
        }
        if extra:
            meta.update(extra)
        self._index[kid] = meta
        self._flush()
        return dict(meta)

    # ---- 读 ----
    def load(self, kid: str) -> dict:
        meta = self._index.get(kid)
        if meta is None:
            raise KeyError(f"关键词不存在: {kid}")
        emb = torch.load(self.store_dir / meta["embedding_file"], map_location="cpu")
        meta = dict(meta)
        meta.update({
            "ea": emb.get("ea"),
            "et": emb.get("et"),
            "em": emb.get("em"),
        })
        return meta

    def list(self) -> list:
        return [dict(self._index[k]) for k in sorted(self._index)]

    def delete(self, kid: str) -> Optional[dict]:
        meta = self._index.pop(kid, None)
        if meta is not None:
            (self.store_dir / meta["embedding_file"]).unlink(missing_ok=True)
            self._flush()
        return meta

    def update_meta(self, kid: str, extra: dict) -> Optional[dict]:
        """更新某关键词的元信息并落盘 (如记录留档音频路径)."""
        if kid not in self._index:
            return None
        self._index[kid].update(extra)
        self._flush()
        return dict(self._index[kid])

    def _flush(self):
        self.index_path.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2), "utf-8")


class KWSEngine:
    """共享模型的关键词引擎: 注册(离线) + 开检测会话(在线).

    注意: decoder 内部模型与注册嵌入是共享可变状态, 注册/切换/前向通过一把
    threading.Lock 串行化. 注册是低频操作, 对在线检测的实时性影响可忽略.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda",
                 store_dir: str = "keyword_store",
                 default_threshold: float = 0.5,
                 ssl_model_dir: Optional[str] = None):
        logger.info(f"加载检查点: {ckpt_path}")
        self.decoder = SynaspotDecoder(ckpt_path, device=device,
                                       ssl_model_dir=ssl_model_dir)
        self.store = KeywordStore(store_dir)
        self.default_threshold = default_threshold
        self.sample_rate = int(self.decoder.cfg.sample_rate)
        self._lock = threading.Lock()
        logger.info(
            f"引擎就绪: backbone={self.decoder.cfg.audio_backbone} "
            f"feat_frame_rate={self.decoder.cfg.feat_frame_rate}Hz "
            f"sample_rate={self.sample_rate} vocab={len(self.decoder.vocab)}")

    @property
    def supports_text(self) -> bool:
        """ssl 主干支持可靠的文本注册; dfsmn 主干 T 模式随机, 建议禁用文本. """
        return getattr(self.decoder.cfg, "audio_backbone", "dfsmn") == "ssl"

    # ---- 注册 (离线) ----
    def register(self, name: str, text: Optional[str] = None,
                 audio_path: Optional[str] = None,
                 threshold: Optional[float] = None,
                 auto_trim: bool = True) -> dict:
        """用文本或音频注册关键词, 提取嵌入并落盘. 至少提供其一.

        auto_trim: 音频注册时自动裁掉首尾静音. 注册模板越紧凑得分越高,
                   因为打分对全部注册单元取几何平均, 且检测滑窗需与注册时长匹配.
        """
        if not text and not audio_path:
            raise ValueError("请提供 text 或 audio 至少一项")
        if text and not self.supports_text:
            raise ValueError(f"当前主干({self.decoder.cfg.audio_backbone}) "
                             f"不支持可靠的文本注册, 请改用 ssl 检查点或使用音频注册")

        enroll_duration = None
        tmp_trim = None
        if audio_path is not None and auto_trim:
            try:
                import soundfile as sf
                from synaspot.dataset import load_wav
                wav = load_wav(audio_path, self.sample_rate)
                trimmed, dur = vad_trim(wav, self.sample_rate)
                if trimmed.numel() > 1600:
                    tmp_trim = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_trim.close()
                    sf.write(tmp_trim.name,
                             trimmed[0].detach().cpu().numpy().astype("float32"),
                             self.sample_rate)
                    audio_path = tmp_trim.name
                    enroll_duration = dur
            except Exception as e:
                logger.warning(f"注册音频静音裁剪失败, 回退原音频: {e}")

        with self._lock:
            self.decoder.enroll(audio_path=audio_path, text=text)
            emb = {
                "ea": self.decoder.ea.cpu() if self.decoder.ea is not None else None,
                "et": self.decoder.et.cpu() if self.decoder.et is not None else None,
                "em": self.decoder.em.cpu() if self.decoder.em is not None else None,
            }
        if emb["ea"] is None and emb["et"] is None:
            raise ValueError("注册失败: 未生成任何嵌入")

        n = int(emb["ea"].shape[0]) if emb["ea"] is not None else 0
        meta = self.store.save(
            name, emb, text=text,
            threshold=threshold if threshold is not None else self.default_threshold,
            audio_name=Path(audio_path).name if audio_path else None,
            extra={"modalities": self._modalities(emb),
                   "enroll_frames": n,
                   "enroll_duration": enroll_duration})

        # 留档注册音频(裁剪后的优先), 便于离线测试与重放
        if audio_path is not None and meta.get("id"):
            try:
                audio_dir = self.store.store_dir / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)
                dest = audio_dir / f"{meta['id']}.wav"
                shutil.copyfile(tmp_trim.name if tmp_trim is not None else audio_path,
                                dest)
                self.store.update_meta(meta["id"], {"audio_path": str(dest)})
                meta["audio_path"] = str(dest)
            except Exception as e:
                logger.warning(f"注册音频留档失败: {e}")

        if tmp_trim is not None:
            Path(tmp_trim.name).unlink(missing_ok=True)

        logger.info(f"注册关键词 [{meta['id']}] {name} "
                    f"modalities={meta['modalities']} threshold={meta['threshold']} "
                    f"注册帧数={n} 时长={enroll_duration or 0:.2f}s")
        return meta

    @staticmethod
    def _modalities(emb: dict) -> list:
        out = []
        if emb["ea"] is not None:
            out.append("A")
        if emb["et"] is not None:
            out.append("T")
        if emb["em"] is not None:
            out.append("M")
        return out

    def list_keywords(self) -> list:
        return self.store.list()

    def delete_keyword(self, kid: str):
        return self.store.delete(kid)

    # ---- 在线检测会话 ----
    def new_session(self, kid: str, window_sec: Optional[float] = None,
                    lookback_frames: Optional[int] = None, cooldown_sec: float = 2.0):
        rec = self.store.load(kid)
        return DetectionSession(self, rec, window_sec=window_sec,
                                lookback_frames=lookback_frames,
                                cooldown_sec=cooldown_sec)


class DetectionSession:
    """一次流式关键词检测会话.

    工作方式: 维护一个固定长度的滑动音频窗口, 每次喂入新 PCM 后对该窗口整体做一次
    前向打分(SSL 主干逐句提特征非因果, 但窗口有界, 延迟=窗口前向耗时, 毫秒级),
    取最近 lookback_frames 帧的峰值得分与阈值比较, 超阈值且超过冷却期即触发.
    """

    def __init__(self, engine: KWSEngine, keyword_rec: dict,
                 window_sec: Optional[float] = None,
                 lookback_frames: Optional[int] = None,
                 cooldown_sec: float = 2.0):
        self.engine = engine
        self.decoder = engine.decoder
        self.keyword = keyword_rec
        self.threshold = float(keyword_rec.get("threshold") or engine.default_threshold)

        # 注册模板规模决定打分窗口与检测滑窗, 两者都需与注册音频匹配
        ea = keyword_rec.get("ea")
        n = int(ea.shape[0]) if ea is not None else 0
        self.enroll_frames = n
        frame_rate = float(getattr(engine.decoder.cfg, "feat_frame_rate", 50) or 50)
        dur = keyword_rec.get("enroll_duration")
        if not dur and n:
            dur = n / frame_rate  # 旧关键词无时长记录时由帧数反推
        self.enroll_duration = float(dur or 1.0)
        # 打分窗口须 >= 注册单元数 n, 否则未对齐单元会把几何平均拖垮
        default_ws = int(getattr(engine.decoder.cfg, "w_scoring", 25) or 25)
        self.w_scoring = max(default_ws, n) if n else default_ws
        # 检测滑窗取注册音频时长: HuBERT 逐句归一化的统计才与注册时一致
        if window_sec is None:
            window_sec = min(max(self.enroll_duration, 0.4), 3.0)
        self.window_sec = float(window_sec)
        self.window_samples = int(engine.sample_rate * self.window_sec)
        if lookback_frames is None:  # 默认取整个窗口内的峰值
            lookback_frames = max(1, int(self.window_sec * frame_rate))
        self.lookback_frames = max(1, int(lookback_frames))
        self.cooldown_sec = cooldown_sec
        self._buf = np.empty(0, dtype=np.float32)
        self._peak = 0.0
        self._last_trigger = -1e9

    def reset(self):
        self._buf = np.empty(0, dtype=np.float32)
        self._peak = 0.0

    def feed_pcm(self, pcm_bytes: bytes) -> tuple:
        """输入 int16 PCM (16kHz 单声道) 字节流, 返回 (score, triggered)."""
        if not pcm_bytes:
            return self._peak, False
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, audio])[-self.window_samples:]
        if self._buf.size < 1600:  # 至少 0.1s
            return 0.0, False

        wav = torch.from_numpy(self._buf.copy()).unsqueeze(0)
        with self.engine._lock:
            dec = self.decoder
            dec.cfg.w_scoring = self.w_scoring  # 打分窗口须覆盖整个注册模板
            dec.reset()
            dec.ea = self.keyword["ea"].to(dec.device) \
                if self.keyword.get("ea") is not None else None
            dec.et = self.keyword["et"].to(dec.device) \
                if self.keyword.get("et") is not None else None
            dec.em = self.keyword["em"].to(dec.device) \
                if self.keyword.get("em") is not None else None
            out = dec.feed(wav)
            score = out.get("score")

        if score is None or score.numel() == 0:
            return self._peak, False

        scores = score.cpu().numpy()
        lb = min(self.lookback_frames, scores.size)
        self._peak = float(scores[-lb:].max()) if lb > 0 else float(scores.max())

        triggered = False
        now = time.time()
        if self._peak >= self.threshold and (now - self._last_trigger) >= self.cooldown_sec:
            triggered = True
            self._last_trigger = now
        return self._peak, triggered


if __name__ == "__main__":
    # 命令行自测: 以文本或音频注册 -> 对一段 wav 模拟流式检测
    import argparse
    import io
    import wave

    ap = argparse.ArgumentParser(description="SYNASPOT KWS 引擎自测")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ssl-model-dir", default=None)
    ap.add_argument("--store-dir", default="keyword_store")
    ap.add_argument("--text", default=None, help="文本关键词, 如 '停一下'")
    ap.add_argument("--audio", default=None, help="注册音频 wav")
    ap.add_argument("--test-wav", default=None, help="用于模拟流式检测的音频")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    engine = KWSEngine(args.ckpt, device=args.device, store_dir=args.store_dir,
                       default_threshold=args.threshold, ssl_model_dir=args.ssl_model_dir)

    if args.text or args.audio:
        kw = engine.register("自测关键词", text=args.text, audio_path=args.audio)
    else:
        kws = engine.list_keywords()
        if not kws:
            raise SystemExit("无已注册关键词, 请用 --text 或 --audio 先注册")
        kw = kws[0]

    print(f"关键词: {kw['name']} id={kw['id']} modalities={kw.get('modalities')} "
          f"threshold={kw['threshold']}")

    if args.test_wav:
        import soundfile as sf
        pcm, sr = sf.read(args.test_wav, dtype="float32")
        assert sr == engine.sample_rate, f"需 {engine.sample_rate}Hz, got {sr}"
        ses = engine.new_session(kw["id"])
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
        raw = buf.getvalue()
        chunk = engine.sample_rate // 10  # 每 0.1s
        peak = 0.0
        trigs = 0
        for i in range(0, len(raw), chunk * 2):
            s, trig = ses.feed_pcm(raw[i:i + chunk * 2])
            peak = max(peak, s)
            trigs += int(trig)
        print(f"峰值得分={peak:.4f} (阈值 {kw['threshold']}), 触发 {trigs} 次")